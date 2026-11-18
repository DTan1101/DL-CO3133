"""Khởi tạo DataLoader cho huấn luyện và cho chuỗi tập test bị che ρ ∈ {...}."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dataset import PetDataset  # noqa: E402
from utils.utils import Config, deep_update, seed_worker  # noqa: E402


def eval_data_cfg(cfg) -> Config:
    """Cấu hình dữ liệu cho đánh giá.

    Giao thức test phải giống nhau giữa mọi thí nghiệm, nên che khuất khi đánh giá
    lấy từ nhánh ``eval.occlusion`` (luôn bật) chứ không lấy theo ``data.occlusion``
    của phương pháp đang huấn luyện: E0 không che khi huấn luyện vẫn phải được đo
    trên cùng các tập test bị che như E6.
    """
    data_cfg = copy.deepcopy(cfg.data.to_dict())
    occlusion = data_cfg.setdefault("occlusion", {})
    deep_update(occlusion, copy.deepcopy(dict(cfg.eval.get("occlusion", {}))))
    occlusion["enabled"] = True
    occlusion["prob"] = 1.0
    data_cfg["occlusion"] = occlusion
    data_cfg["return_masks"] = False
    return Config(data_cfg)


def build_dataset(
    cfg,
    split: str,
    train: Optional[bool] = None,
    rho: Optional[float] = None,
) -> PetDataset:
    is_train = (split == "train") if train is None else bool(train)
    data_cfg = cfg.data if is_train else eval_data_cfg(cfg)
    return PetDataset(
        data_cfg,
        split=split,
        train=train,
        rho=rho,
        eval_seed=int(cfg.data.get("eval_seed", 12345)),
    )


def _loader_kwargs(cfg, train: bool) -> Dict:
    loader_cfg = cfg.data.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", 4))
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(loader_cfg.get("drop_last", True)) and train,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    return kwargs


def build_dataloader(
    cfg,
    split: str,
    train: Optional[bool] = None,
    rho: Optional[float] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    seed: int = 0,
) -> DataLoader:
    dataset = build_dataset(cfg, split, train=train, rho=rho)
    is_train = dataset.train
    loader_cfg = cfg.data.get("loader", {})
    if batch_size is None:
        batch_size = int(
            loader_cfg.get("batch_size", 64)
            if is_train
            else loader_cfg.get("eval_batch_size", loader_cfg.get("batch_size", 64))
        )
    if shuffle is None:
        shuffle = is_train

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        **_loader_kwargs(cfg, is_train),
    )


def build_train_loader(cfg, seed: int = 0) -> DataLoader:
    return build_dataloader(cfg, "train", train=True, seed=seed)


def build_eval_loaders(
    cfg,
    split: str = "test",
    rhos: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> Dict[float, DataLoader]:
    """Một DataLoader cho mỗi mức che ρ, che tất định theo hạt giống cố định.

    ρ = 0 vẫn đi qua nhánh ảnh gốc (không có vật che), nên tương đương ảnh sạch.
    """
    if rhos is None:
        rhos = cfg.eval.get("rhos", [0.0, 0.1, 0.3, 0.5, 0.7])
    loaders: Dict[float, DataLoader] = {}
    for rho in rhos:
        loaders[float(rho)] = build_dataloader(
            cfg, split, train=False, rho=float(rho), shuffle=False, seed=seed
        )
    return loaders


def describe_loaders(loaders: Dict[float, DataLoader]) -> List[str]:
    return [f"ρ={rho:.2f}: {len(loader.dataset)} ảnh" for rho, loader in loaders.items()]
