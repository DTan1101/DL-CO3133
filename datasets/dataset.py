"""Dataset Oxford-IIIT Pet với sinh che khuất trên tiền cảnh và mức nhìn thấy v.

Một mẫu huấn luyện gồm:
  * ``image``      : ảnh gốc sau biến đổi hình học T (đã chuẩn hoá),
  * ``image_occ``  : cùng ảnh đó nhưng bị che một phần tiền cảnh,
  * ``visibility`` : v = sum(m * (1 - o)) / sum(m)  (Định nghĩa 1 trong bản thảo),
  * ``label``      : nhãn giống (0..36).

Phép biến đổi hình học được áp đồng thời lên ảnh và mask tiền cảnh, nên ``image``
và ``image_occ`` chỉ khác nhau ở vùng bị che.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing_data import PetEntry, build_splits, load_split  # noqa: E402
from utils.utils import REPO_ROOT  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Sinh vật che
# --------------------------------------------------------------------------- #
class OcclusionGenerator:
    """Sinh mask vật che o và ảnh bị che x̃ = (1-o)·x + o·η.

    Hai chiến lược:
      * ``foreground_coverage``: thêm vật che cho đến khi tỷ lệ che trên *tiền cảnh*
        đạt ρ (ρ lấy ngẫu nhiên khi huấn luyện, cố định khi đánh giá);
      * ``random_erasing``: xoá một hình chữ nhật ở vị trí bất kỳ trên ảnh theo
        tỷ lệ diện tích *ảnh* (Zhong và cộng sự, 2020) -- dùng cho mốc E1.
    """

    def __init__(self, cfg) -> None:
        self.strategy = str(cfg.get("strategy", "foreground_coverage"))
        self.shape = str(cfg.get("shape", "mixed"))
        self.fill = str(cfg.get("fill", "mean"))
        self.rho_range = tuple(cfg.get("rho_range", (0.0, 0.8)))
        self.prob = float(cfg.get("prob", 1.0))
        self.max_shapes = int(cfg.get("max_shapes", 4))
        self.tolerance = float(cfg.get("tolerance", 0.03))
        self.blob_grid = int(cfg.get("blob_grid", 8))
        self.erase_area = tuple(cfg.get("erase_area", (0.02, 0.4)))
        self.erase_ratio = tuple(cfg.get("erase_ratio", (0.3, 3.3)))
        self.shape_ratio = tuple(cfg.get("shape_ratio", (0.4, 2.5)))

    # -- các dạng vật che -------------------------------------------------- #
    def _rect_mask(
        self,
        shape_hw: Tuple[int, int],
        area_px: float,
        rng: np.random.Generator,
        center: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        height, width = shape_hw
        area_px = float(np.clip(area_px, 4.0, 0.95 * height * width))
        ratio = float(np.exp(rng.uniform(np.log(self.shape_ratio[0]), np.log(self.shape_ratio[1]))))
        box_h = int(round(np.sqrt(area_px * ratio)))
        box_w = int(round(np.sqrt(area_px / ratio)))
        box_h = int(np.clip(box_h, 1, height))
        box_w = int(np.clip(box_w, 1, width))
        if center is None:
            top = int(rng.integers(0, height - box_h + 1))
            left = int(rng.integers(0, width - box_w + 1))
        else:
            top = int(np.clip(center[0] - box_h // 2, 0, height - box_h))
            left = int(np.clip(center[1] - box_w // 2, 0, width - box_w))
        mask = np.zeros(shape_hw, dtype=bool)
        mask[top : top + box_h, left : left + box_w] = True
        return mask

    def _smooth_field(self, shape_hw: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        """Trường ngẫu nhiên tần số thấp, dùng để cắt ra vết che không đều."""
        grid = rng.random((max(2, self.blob_grid), max(2, self.blob_grid))).astype(np.float32)
        field = Image.fromarray(grid, mode="F").resize(
            (shape_hw[1], shape_hw[0]), resample=Image.BILINEAR
        )
        return np.asarray(field, dtype=np.float32)

    def _blob_mask_exact(
        self, fg: np.ndarray, target: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Vết không đều che *đúng* tỷ lệ ``target`` của tiền cảnh (ngưỡng phân vị)."""
        field = self._smooth_field(fg.shape, rng)
        values = field[fg]
        if values.size == 0:
            return np.zeros_like(fg)
        threshold = float(np.quantile(values, min(max(target, 0.0), 1.0)))
        return field <= threshold

    # -- vật liệu tô ------------------------------------------------------- #
    def _fill_values(self, image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Trả về tensor cùng kích thước ảnh, chứa mẫu tô η."""
        if self.fill == "gray":
            return torch.full_like(image, 0.5)
        if self.fill == "mean":
            mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
            return mean.expand_as(image).clone()
        if self.fill == "noise":
            noise = torch.from_numpy(rng.random(tuple(image.shape)).astype(np.float32))
            return noise.to(image.dtype)
        if self.fill == "texture":
            # Lấy kết cấu thật từ chính ảnh bằng cách dịch vòng ảnh đi một lượng ngẫu nhiên.
            height, width = image.shape[-2:]
            shift_h = int(rng.integers(height // 8, max(height // 8 + 1, height - height // 8)))
            shift_w = int(rng.integers(width // 8, max(width // 8 + 1, width - width // 8)))
            return torch.roll(image, shifts=(shift_h, shift_w), dims=(-2, -1))
        raise ValueError(f"Kiểu tô không hỗ trợ: {self.fill}")

    # -- giao diện chính --------------------------------------------------- #
    def __call__(
        self,
        image: torch.Tensor,
        fg: np.ndarray,
        rng: np.random.Generator,
        rho_target: Optional[float] = None,
    ) -> Tuple[torch.Tensor, np.ndarray, float]:
        """Trả về (ảnh bị che, mask vật che, mức nhìn thấy v)."""
        height, width = image.shape[-2:]
        occ = np.zeros((height, width), dtype=bool)

        fg_total = float(fg.sum())
        target_region = fg if fg_total > 0 else np.ones_like(fg)
        region_total = fg_total if fg_total > 0 else float(target_region.sum())

        rho = float(rng.uniform(*self.rho_range)) if rho_target is None else float(rho_target)
        skip = rng.random() > self.prob if rho_target is None else False

        if not skip and rho > 0:
            if self.strategy == "random_erasing":
                occ = self._random_erasing_mask((height, width), rng)
            else:
                occ = self._coverage_mask(target_region, region_total, rho, rng)

        visibility = 1.0
        if fg_total > 0:
            visibility = float((fg & ~occ).sum() / fg_total)

        if occ.any():
            fill = self._fill_values(image, rng)
            occ_t = torch.from_numpy(occ).unsqueeze(0)
            image = torch.where(occ_t, fill, image)
        return image, occ, visibility

    def _random_erasing_mask(
        self, shape_hw: Tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        height, width = shape_hw
        area = float(rng.uniform(*self.erase_area)) * height * width
        ratio = float(np.exp(rng.uniform(np.log(self.erase_ratio[0]), np.log(self.erase_ratio[1]))))
        box_h = int(np.clip(round(np.sqrt(area * ratio)), 1, height))
        box_w = int(np.clip(round(np.sqrt(area / ratio)), 1, width))
        top = int(rng.integers(0, height - box_h + 1))
        left = int(rng.integers(0, width - box_w + 1))
        mask = np.zeros(shape_hw, dtype=bool)
        mask[top : top + box_h, left : left + box_w] = True
        return mask

    def _coverage_mask(
        self,
        region: np.ndarray,
        region_total: float,
        target: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        shape = "blob" if self.shape == "mixed" and rng.random() < 0.5 else self.shape
        if shape == "blob":
            return self._blob_mask_exact(region, target, rng)

        # Dạng chữ nhật: thêm dần từng hình, co lại nếu vượt quá mục tiêu.
        occ = np.zeros_like(region)
        ys, xs = np.nonzero(region)
        for _ in range(self.max_shapes):
            covered = float((region & occ).sum()) / region_total
            remaining = target - covered
            if remaining <= self.tolerance:
                break
            idx = int(rng.integers(0, ys.size)) if ys.size else 0
            center = (int(ys[idx]), int(xs[idx])) if ys.size else None
            # Giả định ~75% diện tích hình rơi vào vùng đích (hình được đặt tâm trên đó).
            area_px = remaining * region_total / 0.75
            candidate = self._rect_mask(region.shape, area_px, rng, center)
            for _ in range(6):
                merged = occ | candidate
                if float((region & merged).sum()) / region_total <= target + self.tolerance:
                    break
                candidate = self._shrink(candidate, 0.75)
            occ |= candidate
        return occ

    @staticmethod
    def _shrink(mask: np.ndarray, factor: float) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return mask
        top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
        box_h = max(1, int(round((bottom - top + 1) * np.sqrt(factor))))
        box_w = max(1, int(round((right - left + 1) * np.sqrt(factor))))
        out = np.zeros_like(mask)
        out[top : top + box_h, left : left + box_w] = True
        return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PetDataset(Dataset):
    """Oxford-IIIT Pet: ảnh + trimap tiền cảnh + sinh che khuất.

    Args:
        cfg: nhánh ``data`` của cấu hình.
        split: ``train`` | ``val`` | ``test``.
        train: True -> biến đổi ngẫu nhiên; False -> resize/center-crop tất định.
        rho: nếu khác None, mọi mẫu bị che đúng mức ρ này (dùng cho test sweep).
        eval_seed: hạt giống cho che khuất tất định khi đánh giá.
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        train: Optional[bool] = None,
        rho: Optional[float] = None,
        eval_seed: int = 12345,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.train = (split == "train") if train is None else bool(train)
        self.rho = rho
        self.eval_seed = int(eval_seed)

        root = Path(cfg.get("root", "datasets/Oxford-IIIT_Pet_Dataset"))
        self.root = root if root.is_absolute() else REPO_ROOT / root
        split_paths = build_splits(
            self.root,
            cfg.get("split_dir", None),
            float(cfg.get("val_ratio", 0.2)),
            int(cfg.get("split_seed", 0)),
            verbose=False,
        )
        self.entries: List[PetEntry] = load_split(split_paths[split])
        self.images_dir = self.root / "images"
        self.trimaps_dir = self.root / "annotations" / "trimaps"

        self.img_size = int(cfg.get("img_size", 224))
        self.fg_values = [int(v) for v in cfg.get("foreground", {}).get("values", [1])]
        self.return_masks = bool(cfg.get("return_masks", False))

        aug = cfg.get("augment", {})
        self.use_rrc = bool(aug.get("random_resized_crop", True)) and self.train
        self.rrc_scale = tuple(aug.get("scale", (0.65, 1.0)))
        self.rrc_ratio = tuple(aug.get("ratio", (3 / 4, 4 / 3)))
        self.hflip_p = float(aug.get("hflip", 0.5)) if self.train else 0.0
        jitter = float(aug.get("color_jitter", 0.0))
        self.color_jitter = (
            transforms.ColorJitter(jitter, jitter, jitter) if (jitter > 0 and self.train) else None
        )
        self.resize_ratio = float(aug.get("resize_ratio", 1.14))

        occ_cfg = cfg.get("occlusion", {})
        self.occlusion_enabled = bool(occ_cfg.get("enabled", True))
        self.occluder = OcclusionGenerator(occ_cfg) if self.occlusion_enabled else None
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def labels(self) -> np.ndarray:
        return np.array([e.label for e in self.entries], dtype=np.int64)

    def class_prior(self, num_classes: int) -> torch.Tensor:
        """Tần suất lớp trên split hiện tại (dùng cho biến thể đích với tiên nghiệm π)."""
        counts = np.bincount(self.labels, minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        return torch.tensor(counts / counts.sum(), dtype=torch.float32)

    # ------------------------------------------------------------------ #
    def _load(self, entry: PetEntry) -> Tuple[Image.Image, Image.Image]:
        image = Image.open(self.images_dir / f"{entry.stem}.jpg").convert("RGB")
        trimap = Image.open(self.trimaps_dir / f"{entry.stem}.png")
        if trimap.mode != "L":
            trimap = trimap.convert("L")
        return image, trimap

    def _geometric(
        self, image: Image.Image, trimap: Image.Image, rng: np.random.Generator
    ) -> Tuple[Image.Image, Image.Image]:
        """Áp cùng một phép biến đổi hình học lên ảnh và mask."""
        size = [self.img_size, self.img_size]
        if self.use_rrc:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image, scale=list(self.rrc_scale), ratio=list(self.rrc_ratio)
            )
            image = TF.resized_crop(
                image, top, left, height, width, size, InterpolationMode.BILINEAR
            )
            trimap = TF.resized_crop(
                trimap, top, left, height, width, size, InterpolationMode.NEAREST
            )
        else:
            short = int(round(self.img_size * self.resize_ratio))
            image = TF.center_crop(
                TF.resize(image, short, InterpolationMode.BILINEAR), size
            )
            trimap = TF.center_crop(
                TF.resize(trimap, short, InterpolationMode.NEAREST), size
            )

        if self.hflip_p > 0 and rng.random() < self.hflip_p:
            image, trimap = TF.hflip(image), TF.hflip(trimap)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        return image, trimap

    def _rng(self, index: int) -> np.random.Generator:
        """Huấn luyện: ngẫu nhiên thật. Đánh giá: tất định theo (eval_seed, index, ρ)."""
        if self.train:
            return np.random.default_rng()
        rho_key = int(round((self.rho or 0.0) * 1000))
        return np.random.default_rng([self.eval_seed, index, rho_key])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        rng = self._rng(index)
        image, trimap = self._load(entry)
        image, trimap = self._geometric(image, trimap, rng)

        trimap_np = np.asarray(trimap, dtype=np.uint8)
        fg = np.isin(trimap_np, self.fg_values)
        clean = TF.to_tensor(image)

        sample: Dict[str, torch.Tensor] = {
            "label": torch.tensor(entry.label, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "fg_ratio": torch.tensor(float(fg.mean()), dtype=torch.float32),
        }

        if self.occluder is not None:
            occluded, occ_mask, visibility = self.occluder(clean, fg, rng, self.rho)
            sample["image_occ"] = self.normalize(occluded)
            sample["visibility"] = torch.tensor(visibility, dtype=torch.float32)
            sample["rho"] = torch.tensor(1.0 - visibility, dtype=torch.float32)
            if self.return_masks:
                sample["mask_fg"] = torch.from_numpy(fg).to(torch.uint8)
                sample["mask_occ"] = torch.from_numpy(occ_mask).to(torch.uint8)
        else:
            sample["visibility"] = torch.tensor(1.0, dtype=torch.float32)
            sample["rho"] = torch.tensor(0.0, dtype=torch.float32)
            if self.return_masks:
                sample["mask_fg"] = torch.from_numpy(fg).to(torch.uint8)
                sample["mask_occ"] = torch.zeros_like(sample["mask_fg"])

        sample["image"] = self.normalize(clean)
        return sample


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Đảo chuẩn hoá về [0,1] (tiện cho việc lưu ảnh kiểm tra)."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def class_names(cfg) -> Sequence[str]:
    from utils.utils import load_json

    root = Path(cfg.get("root", "datasets/Oxford-IIIT_Pet_Dataset"))
    root = root if root.is_absolute() else REPO_ROOT / root
    split_dir = cfg.get("split_dir", None)
    split_dir = Path(split_dir) if split_dir else root / "splits"
    meta_path = split_dir / "meta.json"
    if meta_path.exists():
        return load_json(meta_path).get("class_names", [])
    return []
