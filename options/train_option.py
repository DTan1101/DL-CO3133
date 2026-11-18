"""Tham số dòng lệnh cho huấn luyện: mỗi thí nghiệm là một file YAML trong ``configs/``.

Ví dụ::

    python tools/train.py --config configs/e6_vcc.yaml
    python tools/train.py --config configs/e6_vcc.yaml --seeds 0 1 2
    python tools/train.py --config configs/e6_vcc.yaml --set loss.consistency.lambda=2 model.backbone=resnet50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils import Config, apply_overrides, load_config  # noqa: E402


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Huấn luyện VCC / các mốc so sánh trên Oxford-IIIT Pet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="đường dẫn file YAML của thí nghiệm")
    parser.add_argument("--name", default=None, help="ghi đè tên thí nghiệm (mặc định lấy từ YAML)")
    parser.add_argument("--seed", type=int, default=None, help="hạt giống đơn")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="chạy nhiều hạt giống tuần tự rồi tổng hợp trung bình ± độ lệch chuẩn",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone", default=None, help="ghi đè model.backbone")
    parser.add_argument("--device", default=None, help="auto | cuda | cuda:0 | cpu")
    parser.add_argument("--output-root", default=None, help="thư mục lưu checkpoint và kết quả")
    parser.add_argument("--resume", default=None, help="tiếp tục từ checkpoint")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    parser.add_argument("--workers", type=int, default=None, help="ghi đè data.loader.num_workers")
    parser.add_argument(
        "--eval-split", default=None, choices=["val", "test"],
        help="split dùng cho lượt quét ρ cuối cùng",
    )
    parser.add_argument("--no-final-eval", action="store_true", help="bỏ lượt quét ρ sau huấn luyện")
    parser.add_argument(
        "--debug", action="store_true",
        help="chạy nhanh để kiểm tra pipeline (ít epoch, ít batch mỗi epoch)",
    )
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE",
        help="ghi đè bất kỳ khoá nào trong cấu hình, ví dụ loss.consistency.lambda=2",
    )
    return parser


def _apply_cli(cfg: Config, args: argparse.Namespace) -> Config:
    if args.name is not None:
        cfg.name = args.name
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.loader.batch_size = args.batch_size
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    if args.device is not None:
        cfg.train.device = args.device
    if args.output_root is not None:
        cfg.output.root = args.output_root
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.amp is not None:
        cfg.train.amp = args.amp
    if args.workers is not None:
        cfg.data.loader.num_workers = args.workers
    if args.eval_split is not None:
        cfg.eval.split = args.eval_split
    if args.no_final_eval:
        cfg.eval.final_sweep = False

    if args.seeds:
        cfg.train.seeds = list(args.seeds)
    elif args.seed is not None:
        cfg.train.seeds = [args.seed]
    elif "seeds" not in cfg.train:
        cfg.train.seeds = [int(cfg.train.get("seed", 0))]

    if args.debug:
        cfg.train.epochs = min(int(cfg.train.get("epochs", 1)), 2)
        cfg.train.max_batches_per_epoch = int(cfg.train.get("max_batches_per_epoch", 0) or 4)
        cfg.train.max_eval_batches = int(cfg.train.get("max_eval_batches", 0) or 4)
        cfg.train.seeds = cfg.train.seeds[:1]
        cfg.model.pretrained = cfg.model.get("pretrained", True)
        cfg.name = f"{cfg.get('name', 'exp')}_debug"

    cfg = apply_overrides(cfg, args.overrides)
    return cfg


def parse_train_args(argv: Optional[Sequence[str]] = None) -> Config:
    args = build_train_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = _apply_cli(cfg, args)
    validate_train_config(cfg)
    return cfg


def validate_train_config(cfg: Config) -> None:
    """Kiểm tra sớm những xung đột cấu hình dễ gặp."""
    required = ["name", "data", "model", "train", "loss", "eval"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Cấu hình thiếu các nhánh: {missing}")

    from utils.loss import needs_occluded_branch

    occlusion_on = bool(cfg.data.get("occlusion", {}).get("enabled", True))
    if needs_occluded_branch(cfg) and not occlusion_on:
        raise ValueError(
            "Cấu hình cần nhánh ảnh bị che (consistency hoặc ce_branch=occ/both) "
            "nhưng data.occlusion.enabled=false."
        )

    target_kind = str(cfg.loss.get("consistency", {}).get("target", {}).get("kind", "pred"))
    if cfg.loss.get("visibility_source", "pixel") == "semantic":
        if not cfg.data.get("return_masks", False):
            raise ValueError("visibility_source=semantic yêu cầu data.return_masks=true")
    if target_kind in ("vcc", "soft_aug", "vcc_temp") and not occlusion_on:
        raise ValueError(f"Đích '{target_kind}' cần có che khuất để v < 1.")

    seeds: List[int] = list(cfg.train.get("seeds", [0]))
    if not seeds:
        raise ValueError("Phải có ít nhất một hạt giống.")
