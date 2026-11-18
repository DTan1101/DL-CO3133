"""Tiện ích dùng chung: cấu hình YAML, seed, logger, checkpoint, lịch ramp-up."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
class Config(dict):
    """dict cho phép truy cập bằng thuộc tính: ``cfg.train.epochs``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            super().__setitem__(key, _wrap(value))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - chỉ để thông báo rõ hơn
            raise AttributeError(f"Config không có khoá '{key}'") from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap(value))

    def to_dict(self) -> Dict[str, Any]:
        return _unwrap(self)


def _wrap(value):
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, (list, tuple)):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value):
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap(v) for v in value]
    return value


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Trộn đệ quy ``new`` vào ``base`` (``new`` thắng)."""
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: os.PathLike | str) -> Config:
    """Đọc YAML, hỗ trợ kế thừa qua khoá ``_base_`` (chuỗi hoặc danh sách)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    raw = load_yaml(path)

    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        deep_update(merged, load_config(base_path).to_dict())
    deep_update(merged, raw)

    cfg = Config(merged)
    cfg["config_path"] = str(path)
    return cfg


def dict_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def dict_set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = Config()
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: Config, overrides: Optional[Sequence[str]]) -> Config:
    """Áp các ghi đè dạng ``khoa.con=gia_tri`` (giá trị đọc theo cú pháp YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Ghi đè phải có dạng key=value, nhận được: {item!r}")
        key, raw_value = item.split("=", 1)
        dict_set(cfg, key.strip(), yaml.safe_load(raw_value))
    return cfg


def save_config(cfg: Config, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, allow_unicode=True, sort_keys=False)


# --------------------------------------------------------------------------- #
# Tính tái lập
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Gieo hạt cho worker của DataLoader (mỗi worker một dòng ngẫu nhiên riêng)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Ghi log
# --------------------------------------------------------------------------- #
def get_logger(name: str = "vcc", log_file: Optional[os.PathLike | str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing = {getattr(h, "baseFilename", None) for h in logger.handlers}
        if str(log_file.resolve()) not in existing:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger


class AverageMeter:
    """Trung bình trượt theo số mẫu."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class MeterDict:
    """Nhóm nhiều AverageMeter theo tên."""

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.meters.setdefault(key, AverageMeter()).update(value, n)

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def reset(self) -> None:
        self.start = time.perf_counter()


# --------------------------------------------------------------------------- #
# Lịch trọng số
# --------------------------------------------------------------------------- #
def sigmoid_rampup(current: float, length: float) -> float:
    """Lịch ramp-up kiểu Pi-model (Laine & Aila, 2017)."""
    if length <= 0:
        return 1.0
    phase = 1.0 - min(max(current, 0.0), length) / length
    return float(math.exp(-5.0 * phase * phase))


def linear_rampup(current: float, length: float) -> float:
    if length <= 0:
        return 1.0
    return float(min(max(current, 0.0), length) / length)


def rampup_factor(kind: str, current: float, length: float) -> float:
    kind = (kind or "none").lower()
    if kind in ("none", "const", "constant"):
        return 1.0
    if kind == "sigmoid":
        return sigmoid_rampup(current, length)
    if kind == "linear":
        return linear_rampup(current, length)
    raise ValueError(f"Lịch ramp-up không hỗ trợ: {kind}")


# --------------------------------------------------------------------------- #
# Checkpoint & kết quả
# --------------------------------------------------------------------------- #
def save_checkpoint(state: Dict[str, Any], path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: os.PathLike | str, map_location: Any = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj: Any, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# Khác
# --------------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    parts = []
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if value != 0 and abs(value) < 10 ** (-precision):
            parts.append(f"{key}={value:.2e}")
        else:
            parts.append(f"{key}={value:.{precision}f}")
    return "  ".join(parts)


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)
