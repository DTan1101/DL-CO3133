"""Backbone có thể thay thế: ResNet, ConvNeXt, EfficientNet, ViT, ...

Ưu tiên ``timm`` (hỗ trợ rộng và thống nhất); nếu không có timm thì dùng
``torchvision``. Mọi backbone được bọc lại để có chung giao diện:

    features = backbone.forward_features(x)   # (B, C, h, w) với CNN, (B, N, C) với ViT
    pooled   = backbone.pool(features)        # (B, feat_dim)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:  # pragma: no cover - phụ thuộc môi trường
    import timm

    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False

# Tên trong torchvision tương ứng khi không có timm.
_TORCHVISION_ALIASES = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "convnext_tiny": "convnext_tiny",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3_large_100": "mobilenet_v3_large",
    "vit_base_patch16_224": "vit_b_16",
    "vit_small_patch16_224": None,  # không có trong torchvision
}


class TimmBackbone(nn.Module):
    """Bọc một mô hình timm đã bỏ đầu phân loại (``num_classes=0``)."""

    def __init__(self, name: str, pretrained: bool = True, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=0)
        if drop_path_rate > 0:
            kwargs["drop_path_rate"] = drop_path_rate
        self.model = timm.create_model(name, **kwargs)
        self.name = name
        # Một số mô hình (ví dụ MobileNetV3) có bước chiếu trước đầu phân loại, nên số
        # chiều sau pooling khác num_features; lấy đúng số chiều mà head sẽ nhận.
        self.feat_dim = int(getattr(self.model, "head_hidden_size", None) or self.model.num_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.forward_head(features, pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_features(x))


class TorchvisionBackbone(nn.Module):
    """Phương án dự phòng khi không cài timm."""

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models

        tv_name = _TORCHVISION_ALIASES.get(name, name)
        if tv_name is None:
            raise ValueError(f"Backbone '{name}' cần timm; hãy cài timm hoặc chọn tên khác.")
        if not hasattr(models, tv_name):
            raise ValueError(f"torchvision không có backbone '{tv_name}'.")
        weights = "DEFAULT" if pretrained else None
        model = getattr(models, tv_name)(weights=weights)
        self.name = name
        self.is_transformer = tv_name.startswith("vit")

        if hasattr(model, "fc"):  # họ ResNet
            self.feat_dim = int(model.fc.in_features)
            self.trunk = nn.Sequential(*list(model.children())[:-2])
            self.head_pool = nn.AdaptiveAvgPool2d(1)
        elif hasattr(model, "features"):  # ConvNeXt / EfficientNet / MobileNet
            classifier = model.classifier
            linear = [m for m in classifier.modules() if isinstance(m, nn.Linear)][-1]
            self.feat_dim = int(linear.in_features)
            self.trunk = model.features
            self.head_pool = nn.AdaptiveAvgPool2d(1)
        elif self.is_transformer:
            self.feat_dim = int(model.hidden_dim)
            model.heads = nn.Identity()
            self.trunk = model
            self.head_pool = None
        else:  # pragma: no cover
            raise ValueError(f"Không biết cách bóc đầu phân loại của '{tv_name}'.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        if self.head_pool is None:
            return features
        return torch.flatten(self.head_pool(features), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_features(x))


def create_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
    prefer_timm: bool = True,
) -> Tuple[nn.Module, int]:
    """Trả về (backbone, số chiều đặc trưng)."""
    if prefer_timm and _HAS_TIMM:
        backbone = TimmBackbone(name, pretrained=pretrained, drop_path_rate=drop_path_rate)
    else:
        backbone = TorchvisionBackbone(name, pretrained=pretrained)
    return backbone, backbone.feat_dim


# --------------------------------------------------------------------------- #
# Split BatchNorm (AdvProp / Split-BN): mỗi nhánh trong batch 2B có thống kê riêng
# --------------------------------------------------------------------------- #
class SplitBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm với thống kê riêng cho từng phần bằng nhau của batch.

    Khi huấn luyện, batch ``[x_clean; x_occ]`` được chia thành ``num_splits`` phần;
    phần đầu dùng BN chính, các phần sau dùng BN phụ. Khi suy luận chỉ dùng BN chính
    (vì lúc đó mô hình chỉ nhận một loại ảnh).
    """

    def __init__(self, num_features: int, num_splits: int = 2, **kwargs) -> None:
        super().__init__(num_features, **kwargs)
        if num_splits < 2:
            raise ValueError("num_splits phải >= 2")
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList(
            nn.BatchNorm2d(num_features, **kwargs) for _ in range(num_splits - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(x)
        if x.shape[0] % self.num_splits != 0:
            return super().forward(x)
        chunks = torch.chunk(x, self.num_splits, dim=0)
        outs = [super().forward(chunks[0])]
        outs.extend(bn(chunk) for bn, chunk in zip(self.aux_bn, chunks[1:]))
        return torch.cat(outs, dim=0)


def convert_split_bn(module: nn.Module, num_splits: int = 2) -> nn.Module:
    """Thay mọi BatchNorm2d trong ``module`` bằng SplitBatchNorm2d."""
    if isinstance(module, nn.BatchNorm2d) and not isinstance(module, SplitBatchNorm2d):
        split_bn = SplitBatchNorm2d(
            module.num_features,
            num_splits=num_splits,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            split_bn.weight.data.copy_(module.weight.data)
            split_bn.bias.data.copy_(module.bias.data)
        if module.track_running_stats:
            split_bn.running_mean.copy_(module.running_mean)
            split_bn.running_var.copy_(module.running_var)
            split_bn.num_batches_tracked.copy_(module.num_batches_tracked)
            for bn in split_bn.aux_bn:
                bn.running_mean.copy_(module.running_mean)
                bn.running_var.copy_(module.running_var)
                if module.affine:
                    bn.weight.data.copy_(module.weight.data)
                    bn.bias.data.copy_(module.bias.data)
        return split_bn
    for name, child in module.named_children():
        setattr(module, name, convert_split_bn(child, num_splits))
    return module


def available_backbones(query: Optional[str] = None):
    """Liệt kê tên backbone khả dụng (hữu ích khi chọn cấu hình)."""
    if _HAS_TIMM:
        return timm.list_models(query or "*", pretrained=True)
    return [k for k, v in _TORCHVISION_ALIASES.items() if v is not None]
