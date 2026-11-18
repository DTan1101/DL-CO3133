#!/usr/bin/env python
"""Điểm vào đánh giá: quét các mức che ρ, tính AUC_occ / ECE / RD và xuất bảng.

    python tools/eval.py --checkpoint checkpoints/e6_vcc/seed0/best.pth
    python tools/eval.py --checkpoint ... --baseline checkpoints/e0_baseline/seed0/sweep_test.json
    python tools/eval.py --aggregate "checkpoints/*/seed*/sweep_test.json" --latex
"""
from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options.eval_option import config_from_args, parse_eval_args  # noqa: E402
from trainers.vcc_trainer import VCCTrainer, aggregate_experiment, load_sweep  # noqa: E402
from utils.metrics import (  # noqa: E402
    aggregate_runs,
    summarize_sweep,
    sweep_latex_row,
    sweep_markdown_table,
)
from utils.utils import get_logger, load_checkpoint, save_json  # noqa: E402


def evaluate_checkpoint(args) -> Dict[str, object]:
    cfg = config_from_args(args)
    cfg.eval.final_sweep = False

    logger = get_logger("vcc.eval")
    trainer = VCCTrainer(cfg, seed=int(cfg.train.get("seeds", [0])[0]), inference_only=True)
    state = load_checkpoint(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(state["model"])
    logger.info(f"nạp {args.checkpoint} (epoch {state.get('best_epoch', state.get('epoch', -1)) + 1})")

    split = str(cfg.eval.get("split", "test"))
    sweep = trainer.sweep(split)

    if args.baseline:
        baseline = load_sweep(args.baseline)
        sweep["summary"] = summarize_sweep(sweep["per_rho"], baseline["per_rho"])
        sweep["baseline"] = str(args.baseline)
    if args.save_reliability:
        sweep["reliability"] = _reliability(trainer, cfg, split)

    tag = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / f"sweep_{split}{tag}.json"
    save_json(sweep, out_path)

    logger.info("\n" + sweep_markdown_table({str(cfg.name): sweep}))
    if args.latex:
        logger.info(sweep_latex_row(str(cfg.name), sweep["per_rho"], sweep["summary"]))
    logger.info(f"kết quả -> {out_path}")
    return sweep


def _reliability(trainer: VCCTrainer, cfg, split: str) -> Dict[str, object]:
    """Dữ liệu biểu đồ reliability tại từng mức che (phục vụ phân tích hiệu chỉnh)."""
    from datasets.dataloader import build_eval_loaders
    from utils.metrics import MetricAccumulator

    out: Dict[str, object] = {}
    loaders = build_eval_loaders(cfg, split, cfg.eval.get("rhos", [0.0, 0.7]), seed=trainer.seed)
    for rho, loader in loaders.items():
        accumulator = MetricAccumulator(trainer.num_classes, int(cfg.eval.get("ece_bins", 15)))
        trainer.model.eval()
        import torch

        with torch.no_grad():
            for batch in loader:
                images = batch["image_occ"].to(trainer.device)
                labels = batch["label"].to(trainer.device)
                accumulator.update(trainer.model(images).float(), labels)
        out[f"{rho:g}"] = accumulator.reliability()
    return out


def aggregate_files(patterns: List[str], latex: bool = False) -> Dict[str, object]:
    """Tổng hợp nhiều file sweep_*.json thành bảng mean ± std theo từng thí nghiệm."""
    logger = get_logger("vcc.eval")
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    if not paths:
        raise SystemExit("Không tìm thấy file nào khớp mẫu đã cho.")

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for path in paths:
        sweep = load_sweep(path)
        grouped[str(sweep.get("name", Path(path).parent.parent.name))].append(sweep)

    table: Dict[str, Dict[str, object]] = {}
    for name, sweeps in grouped.items():
        table[name] = aggregate_experiment([{"sweep": s, "cost": {}} for s in sweeps])
        table[name]["summary"] = aggregate_runs([s["summary"] for s in sweeps])

    markdown = sweep_markdown_table(table)
    logger.info("\n" + markdown)
    if latex:
        for name, result in table.items():
            logger.info(sweep_latex_row(name, result["per_rho"], result["summary"]))
    return table


def main() -> None:
    args = parse_eval_args()
    if args.aggregate:
        aggregate_files(args.aggregate, latex=args.latex)
    else:
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
