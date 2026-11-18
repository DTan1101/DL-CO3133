"""Trainer cho khung trọng số--đích thống nhất (E0--E6) trên Oxford-IIIT Pet.

Một bước huấn luyện (Thuật toán 1 của bản thảo):
  1. ảnh gốc và mask chịu cùng phép biến đổi hình học (thực hiện trong Dataset);
  2. sinh vật che trên tiền cảnh, tính mức nhìn thấy v;
  3. ghép [x; x̃] thành batch 2B, truyền một lượt qua cùng mạng f_θ;
  4. L = CE(z, y) + λ(t)·w·D(t ‖ q), với (w, t) do cấu hình quyết định.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dataloader import build_dataset, build_eval_loaders, build_train_loader  # noqa: E402
from models.model import build_model, model_summary, semantic_visibility  # noqa: E402
from utils.loss import build_criterion, needs_occluded_branch  # noqa: E402
from utils.metrics import (  # noqa: E402
    MetricAccumulator,
    aggregate_runs,
    summarize_sweep,
    sweep_latex_row,
    sweep_markdown_table,
)
from utils.utils import (  # noqa: E402
    REPO_ROOT,
    MeterDict,
    format_metrics,
    get_logger,
    load_checkpoint,
    load_json,
    peak_gpu_memory_mb,
    rampup_factor,
    resolve_device,
    save_checkpoint,
    save_config,
    save_json,
    set_seed,
)


class VCCTrainer:
    """Huấn luyện một cấu hình với một hạt giống.

    ``inference_only=True`` chỉ dựng mô hình và đường ống đánh giá (dùng bởi
    ``tools/eval.py``): không tạo loader huấn luyện và không ghi lại config.yaml.
    """

    def __init__(self, cfg, seed: int = 0, inference_only: bool = False) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.seed = int(seed)
        self.inference_only = bool(inference_only)
        self.device = resolve_device(str(self.cfg.train.get("device", "auto")))

        output_root = Path(self.cfg.output.get("root", "checkpoints"))
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        self.out_dir = output_root / str(self.cfg.name) / f"seed{self.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"vcc.{self.cfg.name}.s{self.seed}", self.out_dir / "train.log")

        set_seed(self.seed, bool(self.cfg.train.get("deterministic", False)))

        # --- dữ liệu ---
        self.need_occ = needs_occluded_branch(self.cfg)
        self.need_clean = str(self.cfg.loss.get("ce_branch", "clean")).lower() in ("clean", "both") or bool(
            self.cfg.loss.get("consistency", {}).get("enabled", False)
        )
        self.val_rhos = [float(r) for r in self.cfg.eval.get("val_rhos", [0.0, 0.3, 0.7])]
        if self.inference_only:
            self.train_loader = None
            self.val_loaders = {}
            prior_dataset = build_dataset(self.cfg, "train")
        else:
            self.train_loader = build_train_loader(self.cfg, seed=self.seed)
            self.val_loaders = build_eval_loaders(self.cfg, "val", self.val_rhos, seed=self.seed)
            prior_dataset = self.train_loader.dataset

        # --- mô hình, hàm mục tiêu, tối ưu ---
        self.num_classes = int(self.cfg.data.get("num_classes", 37))
        self.model = build_model(self.cfg).to(self.device)
        prior = prior_dataset.class_prior(self.num_classes)
        self.criterion = build_criterion(self.cfg, class_prior=prior).to(self.device)
        self.visibility_source = str(self.cfg.loss.get("visibility_source", "pixel")).lower()

        self.optimizer = self._build_optimizer()
        self.epochs = int(self.cfg.train.get("epochs", 30))
        self.steps_per_epoch = self._steps_per_epoch()
        self.scheduler = self._build_scheduler()

        self.amp = bool(self.cfg.train.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.grad_clip = float(self.cfg.train.get("grad_clip", 0.0))

        self.monitor = str(self.cfg.train.get("monitor", "top1_mean"))
        self.monitor_mode = str(self.cfg.train.get("monitor_mode", "max"))
        self.best_score = -math.inf if self.monitor_mode == "max" else math.inf
        self.best_epoch = -1
        self.start_epoch = 0
        self.history: List[Dict[str, float]] = []

        self.logger.info(f"[{self.cfg.name} | seed {self.seed}] thiết bị={self.device}")
        self.logger.info(f"mô hình: {model_summary(self.model)}")
        if not self.inference_only:
            save_config(self.cfg, self.out_dir / "config.yaml")
            self.logger.info(
                f"dữ liệu: train={len(self.train_loader.dataset)} "
                f"val={len(self.val_loaders[self.val_rhos[0]].dataset)} "
                f"batch={self.train_loader.batch_size} "
                f"nhánh che={'có' if self.need_occ else 'không'}"
            )
        if self.cfg.train.get("resume"):
            self._resume(self.cfg.train.resume)

    # ------------------------------------------------------------------ #
    # Thiết lập
    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.train.get("optimizer", {})
        name = str(opt_cfg.get("name", "adamw")).lower()
        lr = float(opt_cfg.get("lr", 1e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 0.05))
        groups = self.model.param_groups(
            weight_decay=weight_decay, head_lr_mult=float(opt_cfg.get("head_lr_mult", 1.0)), lr=lr
        )
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=lr, betas=tuple(opt_cfg.get("betas", (0.9, 0.999))))
        if name == "sgd":
            return torch.optim.SGD(
                groups, lr=lr, momentum=float(opt_cfg.get("momentum", 0.9)), nesterov=True
            )
        raise ValueError(f"Bộ tối ưu không hỗ trợ: {name}")

    def _steps_per_epoch(self) -> int:
        if self.train_loader is None:
            return 1
        steps = len(self.train_loader)
        limit = int(self.cfg.train.get("max_batches_per_epoch", 0) or 0)
        return min(steps, limit) if limit > 0 else steps

    def _build_scheduler(self):
        sched_cfg = self.cfg.train.get("scheduler", {})
        kind = str(sched_cfg.get("name", "cosine")).lower()
        total = max(1, self.epochs * self.steps_per_epoch)
        warmup = int(float(sched_cfg.get("warmup_epochs", 1)) * self.steps_per_epoch)
        min_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

        if kind in ("none", "const", "constant"):
            return None

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            if kind == "cosine":
                progress = (step - warmup) / max(1, total - warmup)
                progress = min(max(progress, 0.0), 1.0)
                return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            if kind == "step":
                gamma = float(sched_cfg.get("gamma", 0.1))
                milestones = [
                    int(float(m) * self.steps_per_epoch) for m in sched_cfg.get("milestones", [])
                ]
                return gamma ** sum(step >= m for m in milestones)
            raise ValueError(f"Lịch học không hỗ trợ: {kind}")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _resume(self, path: str | Path) -> None:
        state = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = int(state.get("epoch", -1)) + 1
        self.best_score = float(state.get("best_score", self.best_score))
        self.history = list(state.get("history", []))
        self.logger.info(f"tiếp tục từ {path} (epoch {self.start_epoch})")

    # ------------------------------------------------------------------ #
    # Một epoch
    # ------------------------------------------------------------------ #
    def _lambda_scale(self, epoch: int, step: int) -> float:
        ramp = self.cfg.train.get("rampup", {})
        progress = epoch + (step / max(1, self.steps_per_epoch))
        return rampup_factor(
            str(ramp.get("kind", "sigmoid")), progress, float(ramp.get("epochs", 5))
        )

    def _forward(self, batch, labels: torch.Tensor):
        """Truyền một lượt, trả về (logits ảnh gốc, logits ảnh che, mức nhìn thấy)."""
        images = batch["image"].to(self.device, non_blocking=True)
        need_map = self.visibility_source == "semantic"

        if not self.need_occ:
            logits = self.model(images)
            return logits.float(), None, None

        occluded = batch["image_occ"].to(self.device, non_blocking=True)
        visibility = batch["visibility"].to(self.device, non_blocking=True)

        if not self.need_clean:  # E1/E2: chỉ huấn luyện trên ảnh bị che
            logits_occ = self.model(occluded)
            return None, logits_occ.float(), visibility

        batch_size = images.shape[0]
        inputs = torch.cat([images, occluded], dim=0)
        outputs = self.model(inputs, return_map=need_map)
        if need_map:
            logits, feature_map = outputs
            visibility = self._semantic_visibility(
                feature_map, labels, batch, batch_size, fallback=visibility
            )
        else:
            logits = outputs
        logits = logits.float()
        return logits[:batch_size], logits[batch_size:], visibility

    def _semantic_visibility(
        self,
        feature_map: Optional[torch.Tensor],
        labels: torch.Tensor,
        batch,
        batch_size: int,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """v_sem có trọng số CAM (Mục 4.6): đo thông tin mất đi thay vì diện tích mất đi."""
        if feature_map is None or "mask_fg" not in batch:
            if not getattr(self, "_warned_semantic", False):
                self.logger.info(
                    "visibility_source=semantic nhưng backbone không cho bản đồ đặc trưng 4 chiều "
                    "(hoặc dataset không trả mask) -> dùng mức nhìn thấy theo pixel."
                )
                self._warned_semantic = True
            return fallback
        with torch.no_grad():
            cam = self.model.class_activation_map(feature_map[:batch_size].float(), labels)
            mask_fg = batch["mask_fg"][:batch_size].to(self.device, non_blocking=True)
            mask_occ = batch["mask_occ"][:batch_size].to(self.device, non_blocking=True)
            return semantic_visibility(cam, mask_fg, mask_occ)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = MeterDict()
        start = time.perf_counter()
        log_every = int(self.cfg.train.get("log_every", 50))

        for step, batch in enumerate(self.train_loader):
            if step >= self.steps_per_epoch:
                break
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.amp):
                logits_clean, logits_occ, visibility = self._forward(batch, labels)
            loss, stats = self.criterion(
                logits_clean, logits_occ, labels, visibility,
                lambda_scale=self._lambda_scale(epoch, step),
            )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            meters.update(stats, n=labels.size(0))
            if log_every > 0 and step % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"epoch {epoch + 1}/{self.epochs} step {step}/{self.steps_per_epoch} "
                    f"lr={lr:.2e} {format_metrics(stats)}"
                )

        out = meters.averages()
        out["epoch_time_s"] = time.perf_counter() - start
        out["lr"] = self.optimizer.param_groups[0]["lr"]
        return out

    # ------------------------------------------------------------------ #
    # Đánh giá
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate_loader(self, loader, max_batches: int = 0) -> Dict[str, float]:
        """Đánh giá trên một tập đã cố định mức che: mô hình chỉ nhận ảnh (có thể bị che)."""
        self.model.eval()
        accumulator = MetricAccumulator(self.num_classes, int(self.cfg.eval.get("ece_bins", 15)))
        use_occ = loader.dataset.occluder is not None

        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            key = "image_occ" if use_occ else "image"
            images = batch[key].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                logits = self.model(images)
            accumulator.update(logits.float(), labels, batch.get("visibility"))
        return accumulator.compute()

    @torch.no_grad()
    def sweep(
        self,
        split: str = "test",
        rhos: Optional[Sequence[float]] = None,
        loaders=None,
        max_batches: int = 0,
    ) -> Dict[str, object]:
        """Quét các mức che ρ và tổng hợp AUC_occ."""
        if loaders is None:
            rhos = rhos if rhos is not None else [float(r) for r in self.cfg.eval.get("rhos", [0.0, 0.1, 0.3, 0.5, 0.7])]
            loaders = build_eval_loaders(self.cfg, split, rhos, seed=self.seed)
        per_rho = {
            float(rho): self.evaluate_loader(loader, max_batches)
            for rho, loader in loaders.items()
        }
        return {
            "name": str(self.cfg.name),
            "seed": self.seed,
            "split": split,
            "per_rho": per_rho,
            "summary": summarize_sweep(per_rho),
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        result = self.sweep(
            "val",
            loaders=self.val_loaders,
            max_batches=int(self.cfg.train.get("max_eval_batches", 0) or 0),
        )
        flat: Dict[str, float] = {}
        for rho, metrics in result["per_rho"].items():
            for key, value in metrics.items():
                flat[f"{key}@{rho:g}"] = value
        flat.update(result["summary"])
        return flat

    # ------------------------------------------------------------------ #
    # Vòng huấn luyện
    # ------------------------------------------------------------------ #
    def _is_better(self, score: float) -> bool:
        if self.monitor_mode == "max":
            return score > self.best_score
        return score < self.best_score

    def _save(self, epoch: int, tag: str) -> None:
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "scaler": self.scaler.state_dict() if self.amp else None,
                "epoch": epoch,
                "seed": self.seed,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "monitor": self.monitor,
                "history": self.history,
                "config": self.cfg.to_dict(),
            },
            self.out_dir / f"{tag}.pth",
        )

    def fit(self) -> Dict[str, object]:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        patience = int(self.cfg.train.get("early_stop_patience", 0) or 0)
        eval_every = int(self.cfg.train.get("eval_every", 1))
        bad_epochs = 0

        for epoch in range(self.start_epoch, self.epochs):
            train_stats = self.train_one_epoch(epoch)
            record = {"epoch": epoch + 1, **{f"train/{k}": v for k, v in train_stats.items()}}

            if (epoch + 1) % eval_every == 0 or epoch + 1 == self.epochs:
                val_stats = self.validate(epoch)
                record.update({f"val/{k}": v for k, v in val_stats.items()})
                score = float(val_stats.get(self.monitor, val_stats.get("top1_mean", 0.0)))
                improved = self._is_better(score)
                if improved:
                    self.best_score, self.best_epoch, bad_epochs = score, epoch, 0
                    self._save(epoch, "best")
                else:
                    bad_epochs += 1
                self.logger.info(
                    f"epoch {epoch + 1}/{self.epochs} | train {format_metrics(train_stats, 3)} | "
                    f"val {self.monitor}={score:.3f}"
                    f"{' (tốt nhất)' if improved else ''} | {train_stats['epoch_time_s']:.1f}s"
                )
                self.logger.info("  " + format_metrics(val_stats, 3))
            else:
                self.logger.info(
                    f"epoch {epoch + 1}/{self.epochs} | train {format_metrics(train_stats, 3)}"
                )

            self.history.append(record)
            self._save(epoch, "last")
            save_json(self.history, self.out_dir / "history.json")
            if patience and bad_epochs >= patience:
                self.logger.info(f"dừng sớm sau {bad_epochs} epoch không cải thiện")
                break

        cost = {
            "epoch_time_s_mean": float(
                sum(r.get("train/epoch_time_s", 0.0) for r in self.history) / max(1, len(self.history))
            ),
            "peak_gpu_mem_MB": peak_gpu_memory_mb(),
            "params": model_summary(self.model),
        }
        save_json(cost, self.out_dir / "cost.json")

        result: Dict[str, object] = {
            "name": str(self.cfg.name),
            "seed": self.seed,
            "best_epoch": self.best_epoch + 1,
            "best_score": self.best_score,
            "monitor": self.monitor,
            "cost": cost,
            "out_dir": str(self.out_dir),
        }

        if bool(self.cfg.eval.get("final_sweep", True)):
            self._load_best()
            split = str(self.cfg.eval.get("split", "test"))
            sweep = self.sweep(split, max_batches=int(self.cfg.train.get("max_eval_batches", 0) or 0))
            save_json(sweep, self.out_dir / f"sweep_{split}.json")
            result["sweep"] = sweep
            self.logger.info(f"[{self.cfg.name} | seed {self.seed}] kết quả {split}:")
            self.logger.info(
                "\n" + sweep_markdown_table({str(self.cfg.name): sweep})
            )
        save_json(result, self.out_dir / "result.json")
        return result

    def _load_best(self) -> None:
        best_path = self.out_dir / "best.pth"
        if best_path.exists():
            state = load_checkpoint(best_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.logger.info(f"nạp lại checkpoint tốt nhất (epoch {state.get('best_epoch', -1) + 1})")


# --------------------------------------------------------------------------- #
# Chạy một thí nghiệm với nhiều hạt giống
# --------------------------------------------------------------------------- #
def run_experiment(cfg) -> Dict[str, object]:
    """Huấn luyện tuần tự theo từng hạt giống rồi tổng hợp trung bình ± độ lệch chuẩn."""
    seeds = [int(s) for s in cfg.train.get("seeds", [0])]
    logger = get_logger("vcc.runner")
    results: List[Dict[str, object]] = []

    for run_index, seed in enumerate(seeds, start=1):
        logger.info("=" * 78)
        logger.info(f"THÍ NGHIỆM {cfg.name} | hạt giống {seed} ({run_index}/{len(seeds)})")
        logger.info("=" * 78)
        results.append(VCCTrainer(cfg, seed=seed).fit())

    output_root = Path(cfg.output.get("root", "checkpoints"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    exp_dir = output_root / str(cfg.name)
    summary = aggregate_experiment(results)
    if summary:
        save_json(summary, exp_dir / "summary.json")
        table = sweep_markdown_table({str(cfg.name): summary})
        (exp_dir / "summary.md").write_text(
            f"# {cfg.name}\n\n{table}\n\n"
            f"LaTeX: `{sweep_latex_row(str(cfg.name), summary['per_rho'], summary['summary'])}`\n",
            encoding="utf-8",
        )
        logger.info(f"tổng hợp {len(results)} hạt giống -> {exp_dir / 'summary.json'}")
        logger.info("\n" + table)
    return {"name": str(cfg.name), "runs": results, "summary": summary}


def aggregate_experiment(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Gộp các lượt quét ρ của nhiều hạt giống thành mean/std theo từng ρ."""
    sweeps = [r["sweep"] for r in results if "sweep" in r]
    if not sweeps:
        return {}
    rhos = sorted({float(rho) for sweep in sweeps for rho in sweep["per_rho"]})
    per_rho = {
        rho: aggregate_runs([sweep["per_rho"][rho] for sweep in sweeps if rho in sweep["per_rho"]])
        for rho in rhos
    }
    return {
        "name": sweeps[0]["name"],
        "split": sweeps[0]["split"],
        "seeds": [sweep["seed"] for sweep in sweeps],
        "per_rho": per_rho,
        "summary": aggregate_runs([sweep["summary"] for sweep in sweeps]),
        "cost": aggregate_runs(
            [{k: v for k, v in r["cost"].items() if isinstance(v, (int, float))} for r in results]
        ),
    }


def load_sweep(path: str | Path) -> Dict[str, object]:
    """Đọc lại file ``sweep_*.json`` (khoá ρ trong JSON là chuỗi -> đổi về float)."""
    data = load_json(path)
    data["per_rho"] = {float(k): v for k, v in data["per_rho"].items()}
    return data
