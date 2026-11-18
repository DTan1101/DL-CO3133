#!/usr/bin/env python
"""Điểm vào huấn luyện: mỗi thí nghiệm là một file YAML trong ``configs/``.

    python tools/train.py --config configs/e6_vcc.yaml --seeds 0 1 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options.train_option import parse_train_args  # noqa: E402
from trainers.vcc_trainer import run_experiment  # noqa: E402


def main() -> None:
    cfg = parse_train_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
