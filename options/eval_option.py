"""Tham số dòng lệnh cho đánh giá: quét ρ ∈ {0; 0,1; 0,3; 0,5; 0,7} trên tập test.

Ví dụ::

    python tools/eval.py --checkpoint checkpoints/e6_vcc/seed0/best.pth
    python tools/eval.py --config configs/e6_vcc.yaml --checkpoint .../best.pth \
        --baseline checkpoints/e0_baseline/seed0/sweep_test.json
    python tools/eval.py --aggregate "checkpoints/e6_vcc/seed*/sweep_test.json"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils import Config, apply_overrides, load_config  # noqa: E402


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình trên các tập test bị che có kiểm soát",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default=None, help="mặc định lấy cấu hình lưu trong checkpoint")
    parser.add_argument("--checkpoint", "-k", default=None, help="đường dẫn .pth")
    parser.add_argument("--split", default=None, choices=["val", "test"])
    parser.add_argument("--rhos", type=float, nargs="+", default=None, help="danh sách mức che")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--baseline", default=None, help="file sweep_*.json của E0 để tính RD")
    parser.add_argument("--out", default=None, help="file JSON kết quả (mặc định cạnh checkpoint)")
    parser.add_argument("--tag", default=None, help="hậu tố cho tên file kết quả")
    parser.add_argument("--save-reliability", action="store_true", help="lưu dữ liệu biểu đồ reliability")
    parser.add_argument(
        "--aggregate", nargs="+", default=None,
        help="tổng hợp nhiều file sweep_*.json (hỗ trợ ký tự thay thế) thành bảng mean ± std",
    )
    parser.add_argument("--latex", action="store_true", help="in thêm dòng LaTeX cho Bảng kết quả")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE")
    return parser


def parse_eval_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_eval_parser().parse_args(argv)
    if args.aggregate is None and args.checkpoint is None:
        raise SystemExit("Cần --checkpoint (hoặc --aggregate để tổng hợp kết quả đã có).")
    return args


def config_from_args(args: argparse.Namespace) -> Config:
    """Lấy cấu hình: ưu tiên --config, nếu không thì đọc từ checkpoint."""
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        import torch

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" not in state:
            raise SystemExit("Checkpoint không chứa cấu hình; hãy truyền --config.")
        cfg = Config(state["config"])

    if args.split is not None:
        cfg.eval.split = args.split
    if args.rhos is not None:
        cfg.eval.rhos = list(args.rhos)
    if args.device is not None:
        cfg.train.device = args.device
    if args.batch_size is not None:
        cfg.data.loader.eval_batch_size = args.batch_size
    if args.workers is not None:
        cfg.data.loader.num_workers = args.workers
    return apply_overrides(cfg, args.overrides)
