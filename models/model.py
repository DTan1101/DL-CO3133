"""Mô hình phân loại: backbone + đầu tuyến tính, kèm CAM cho mức nhìn thấy ngữ nghĩa."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.backbone import convert_split_bn, create_backbone  # noqa: E402


class PetClassifier(nn.Module):
    """f_θ: ảnh -> logits z ∈ R^C.

    Mạng dùng chung cho hai nhánh (ảnh gốc và ảnh bị che); batch 2B được truyền
    một lượt nên BatchNorm dùng chung thống kê, trừ khi bật ``split_bn``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 37,
        pretrained: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        split_bn: bool = False,
        num_splits: int = 2,
        prefer_timm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, self.feat_dim = create_backbone(
            backbone, pretrained=pretrained, drop_path_rate=drop_path_rate, prefer_timm=prefer_timm
        )
        self.backbone_name = backbone
        self.num_classes = int(num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.feat_dim, self.num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        self.split_bn = bool(split_bn)
        if self.split_bn:
            convert_split_bn(self.backbone, num_splits=num_splits)

    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor, return_map: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.backbone.forward_features(x)
        pooled = self.backbone.pool(features)
        logits = self.head(self.dropout(pooled))
        if not return_map:
            return logits
        feature_map = features if features.dim() == 4 else None
        return logits, feature_map

    @torch.no_grad()
    def class_activation_map(
        self, feature_map: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """CAM kiểu GAP: a = ReLU(Σ_c w[y,c]·F_c), chuẩn hoá về [0,1] theo từng mẫu.

        Dùng cho mức nhìn thấy ngữ nghĩa v_sem (Mục 4.6 của bản thảo). Chỉ hợp lệ
        với backbone cho bản đồ đặc trưng 4 chiều.
        """
        if feature_map is None or feature_map.dim() != 4:
            raise ValueError("CAM cần bản đồ đặc trưng (B, C, h, w) từ backbone dạng CNN.")
        weight = self.head.weight.index_select(0, labels)  # (B, feat_dim)
        if weight.shape[1] != feature_map.shape[1]:
            # Một số backbone có bước chiếu trước đầu phân loại -> dùng năng lượng kênh.
            cam = feature_map.clamp(min=0).mean(dim=1)
        else:
            cam = torch.einsum("bc,bchw->bhw", weight, feature_map).clamp(min=0)
        flat = cam.flatten(1)
        maximum = flat.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        return (flat / maximum).view_as(cam)

    # ------------------------------------------------------------------ #
    def param_groups(self, weight_decay: float = 0.05, head_lr_mult: float = 1.0, lr: float = 1e-4):
        """Không áp weight decay lên bias và tham số chuẩn hoá; đầu phân loại có thể có lr riêng."""
        decay, no_decay, head_decay, head_no_decay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("head.")
            skip = param.ndim <= 1 or name.endswith(".bias")
            if is_head:
                (head_no_decay if skip else head_decay).append(param)
            else:
                (no_decay if skip else decay).append(param)
        groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
            {"params": head_decay, "weight_decay": weight_decay, "lr": lr * head_lr_mult},
            {"params": head_no_decay, "weight_decay": 0.0, "lr": lr * head_lr_mult},
        ]
        return [g for g in groups if g["params"]]


def build_model(cfg) -> PetClassifier:
    model_cfg = cfg.model
    return PetClassifier(
        backbone=str(model_cfg.get("backbone", "resnet18")),
        num_classes=int(cfg.data.get("num_classes", 37)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        split_bn=bool(model_cfg.get("split_bn", False)),
        num_splits=int(model_cfg.get("num_splits", 2)),
        prefer_timm=bool(model_cfg.get("prefer_timm", True)),
    )


@torch.no_grad()
def semantic_visibility(
    cam: torch.Tensor, mask_fg: torch.Tensor, mask_occ: torch.Tensor
) -> torch.Tensor:
    """v_sem = Σ m·(1-o)·a / Σ m·a, với a là CAM trên ảnh gốc.

    CAM ở độ phân giải thấp nên mask được hạ mẫu về cùng kích thước.
    """
    cam = cam.unsqueeze(1).float()
    size = cam.shape[-2:]
    fg = F.adaptive_avg_pool2d(mask_fg.unsqueeze(1).float(), size)
    occ = F.adaptive_avg_pool2d(mask_occ.unsqueeze(1).float(), size)
    numerator = (fg * (1.0 - occ) * cam).flatten(1).sum(1)
    denominator = (fg * cam).flatten(1).sum(1)
    visibility = numerator / denominator.clamp(min=1e-6)
    # Mẫu không có tiền cảnh -> coi như nhìn thấy đầy đủ.
    return torch.where(denominator > 1e-6, visibility, torch.ones_like(visibility)).clamp(0, 1)


def model_summary(model: PetClassifier) -> Dict[str, object]:
    from utils.utils import count_parameters

    params = count_parameters(model)
    return {
        "backbone": model.backbone_name,
        "feat_dim": model.feat_dim,
        "num_classes": model.num_classes,
        "split_bn": model.split_bn,
        "params_total_M": round(params["total"] / 1e6, 2),
        "params_trainable_M": round(params["trainable"] / 1e6, 2),
    }
