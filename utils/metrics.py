"""Độ đo: Top-1/Top-5, Macro-F1, ECE (15 khoảng), AUC_occ và độ suy giảm tương đối.

Không dùng scikit-learn để giảm phụ thuộc: Macro-F1 tính từ ma trận nhầm lẫn.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Độ chính xác
# --------------------------------------------------------------------------- #
def topk_correct(logits: torch.Tensor, labels: torch.Tensor, ks: Sequence[int] = (1, 5)) -> Dict[int, int]:
    """Số mẫu đúng ở Top-k cho từng k."""
    max_k = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    hits = pred.eq(labels.view(-1, 1))
    return {k: int(hits[:, : min(k, max_k)].any(dim=1).sum()) for k in ks}


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = labels.to(torch.long) * num_classes + preds.to(torch.long)
    counts = torch.bincount(idx, minlength=num_classes**2)
    return counts.view(num_classes, num_classes)


def macro_f1_from_confusion(matrix: torch.Tensor) -> float:
    """Macro-F1: trung bình không trọng số của F1 trên từng lớp."""
    matrix = matrix.to(torch.float64)
    tp = matrix.diag()
    fp = matrix.sum(dim=0) - tp
    fn = matrix.sum(dim=1) - tp
    precision = tp / (tp + fp).clamp(min=1e-12)
    recall = tp / (tp + fn).clamp(min=1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = (matrix.sum(dim=1) > 0)
    if present.sum() == 0:
        return float("nan")
    return float(f1[present].mean())


# --------------------------------------------------------------------------- #
# Hiệu chỉnh
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15
) -> float:
    """ECE với các khoảng đều nhau (Guo và cộng sự, 2017)."""
    confidence = confidence.to(torch.float64)
    correct = correct.to(torch.float64)
    bins = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    ece = 0.0
    total = confidence.numel()
    if total == 0:
        return float("nan")
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        accuracy = float(correct[in_bin].mean())
        avg_conf = float(confidence[in_bin].mean())
        ece += (count / total) * abs(avg_conf - accuracy)
    return float(ece)


def reliability_bins(
    confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15
) -> List[Dict[str, float]]:
    """Dữ liệu cho biểu đồ reliability (độ tin cậy vs độ chính xác theo khoảng)."""
    bins = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    out = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        out.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": float(correct[in_bin].to(torch.float64).mean()) if count else float("nan"),
                "confidence": float(confidence[in_bin].to(torch.float64).mean()) if count else float("nan"),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Bộ tích luỹ cho một lượt đánh giá
# --------------------------------------------------------------------------- #
class MetricAccumulator:
    """Gom logits/nhãn theo từng batch rồi tính toàn bộ độ đo một lần."""

    def __init__(self, num_classes: int, n_bins: int = 15) -> None:
        self.num_classes = int(num_classes)
        self.n_bins = int(n_bins)
        self._logits: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._visibility: List[torch.Tensor] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> None:
        self._logits.append(logits.detach().float().cpu())
        self._labels.append(labels.detach().cpu())
        if visibility is not None:
            self._visibility.append(visibility.detach().float().cpu())

    @property
    def empty(self) -> bool:
        return not self._logits

    def compute(self) -> Dict[str, float]:
        if self.empty:
            return {}
        logits = torch.cat(self._logits)
        labels = torch.cat(self._labels)
        probs = F.softmax(logits, dim=1)
        confidence, preds = probs.max(dim=1)
        correct = preds.eq(labels)

        hits = topk_correct(logits, labels, ks=(1, 5))
        total = labels.numel()
        matrix = confusion_matrix(preds, labels, self.num_classes)

        metrics = {
            "top1": 100.0 * hits[1] / total,
            "top5": 100.0 * hits[5] / total,
            "macro_f1": 100.0 * macro_f1_from_confusion(matrix),
            "ece": expected_calibration_error(confidence, correct, self.n_bins),
            "nll": float(F.cross_entropy(logits, labels).item()),
            "brier": float(
                (probs - F.one_hot(labels, self.num_classes).float()).pow(2).sum(dim=1).mean()
            ),
            "mean_confidence": float(confidence.mean()),
            "accuracy": float(correct.float().mean()),
            "n": int(total),
        }
        if self._visibility:
            metrics["mean_visibility"] = float(torch.cat(self._visibility).mean())
        return metrics

    def reliability(self) -> List[Dict[str, float]]:
        if self.empty:
            return []
        probs = F.softmax(torch.cat(self._logits), dim=1)
        labels = torch.cat(self._labels)
        confidence, preds = probs.max(dim=1)
        return reliability_bins(confidence, preds.eq(labels), self.n_bins)


# --------------------------------------------------------------------------- #
# Độ đo tổng hợp trên chuỗi mức che
# --------------------------------------------------------------------------- #
def auc_occ(rhos: Sequence[float], accuracies: Sequence[float]) -> float:
    """Diện tích dưới đường accuracy--ρ, chuẩn hoá về [0,1].

    ``accuracies`` tính theo phần trăm; kết quả cũng theo phần trăm (trung bình
    có trọng số theo khoảng ρ bằng phép hình thang).
    """
    order = np.argsort(np.asarray(rhos, dtype=np.float64))
    x = np.asarray(rhos, dtype=np.float64)[order]
    y = np.asarray(accuracies, dtype=np.float64)[order]
    if x.size < 2:
        return float(y[0]) if y.size else float("nan")
    span = x[-1] - x[0]
    if span <= 0:
        return float(y.mean())
    return float(np.trapezoid(y, x) / span) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / span)


def relative_degradation(
    err_method: float, err_method_clean: float, err_base: float, err_base_clean: float
) -> float:
    """RD(ρ) = [Err_f(ρ) - Err_f(0)] / [Err_E0(ρ) - Err_E0(0)] (Hendrycks & Dietterich)."""
    denominator = err_base - err_base_clean
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((err_method - err_method_clean) / denominator)


def summarize_sweep(
    per_rho: Dict[float, Dict[str, float]],
    baseline: Optional[Dict[float, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Tổng hợp một lượt quét ρ: AUC_occ, Top-1 trung bình, RD theo từng ρ."""
    rhos = sorted(per_rho)
    accuracies = [per_rho[r]["top1"] for r in rhos]
    summary: Dict[str, float] = {
        "auc_occ": auc_occ(rhos, accuracies),
        "top1_mean": float(np.mean(accuracies)),
        "top1_clean": per_rho[rhos[0]]["top1"] if rhos else float("nan"),
        "top1_worst": float(np.min(accuracies)) if accuracies else float("nan"),
    }
    if baseline:
        clean_rho = min(rhos)
        err_clean = 100.0 - per_rho[clean_rho]["top1"]
        base_err_clean = 100.0 - baseline[clean_rho]["top1"]
        for rho in rhos:
            if rho == clean_rho or rho not in baseline:
                continue
            summary[f"rd@{rho:g}"] = relative_degradation(
                100.0 - per_rho[rho]["top1"], err_clean,
                100.0 - baseline[rho]["top1"], base_err_clean,
            )
    return summary


def aggregate_runs(runs: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Trung bình ± độ lệch chuẩn của cùng một độ đo qua nhiều hạt giống."""
    keys = sorted({k for run in runs for k, v in run.items() if isinstance(v, (int, float))})
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [float(run[key]) for run in runs if key in run and run[key] is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}
    return out


def format_mean_std(stats: Dict[str, float], precision: int = 2) -> str:
    return f"{stats['mean']:.{precision}f} ± {stats['std']:.{precision}f}"


# --------------------------------------------------------------------------- #
# Trình bày kết quả
# --------------------------------------------------------------------------- #
_REPORT_RHOS = (0.0, 0.3, 0.7)


def sweep_markdown_table(
    runs: Dict[str, Dict[str, object]], report_rhos: Sequence[float] = _REPORT_RHOS
) -> str:
    """Bảng markdown theo đúng các cột của Bảng kết quả trong bản thảo.

    ``runs``: {tên thí nghiệm: {"per_rho": {ρ: {...}}, "summary": {...}}}, trong đó
    mỗi giá trị có thể là số (một hạt giống) hoặc dict mean/std (nhiều hạt giống).
    """
    def cell(value) -> str:
        if value is None:
            return "---"
        if isinstance(value, dict) and "mean" in value:
            return format_mean_std(value)
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)

    header = ["Mã"] + [f"Top-1 (ρ={r:g})" for r in report_rhos] + [
        "AUC_occ", "ECE (ρ=0,7)", "Top-5 (ρ=0,7)"
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, result in runs.items():
        per_rho = result.get("per_rho", {})
        summary = result.get("summary", {})
        row = [name]
        for rho in report_rhos:
            row.append(cell(_lookup(per_rho, rho, "top1")))
        row.append(cell(summary.get("auc_occ")))
        row.append(cell(_lookup(per_rho, 0.7, "ece", scale=1.0)))
        row.append(cell(_lookup(per_rho, 0.7, "top5")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def sweep_latex_row(
    name: str,
    per_rho: Dict[object, Dict[str, object]],
    summary: Dict[str, object],
    report_rhos: Sequence[float] = _REPORT_RHOS,
) -> str:
    """Một dòng LaTeX để dán trực tiếp vào bảng kết quả của bản thảo."""
    def cell(value) -> str:
        if value is None:
            return "---"
        if isinstance(value, dict) and "mean" in value:
            return f"{value['mean']:.2f}$\\pm${value['std']:.2f}"
        return f"{float(value):.2f}"

    cells = [cell(_lookup(per_rho, r, "top1")) for r in report_rhos]
    cells.append(cell(summary.get("auc_occ")))
    cells.append(cell(_lookup(per_rho, 0.7, "ece")))
    cells.append(cell(_lookup(per_rho, 0.7, "top5")))
    return f"{name} & " + " & ".join(cells) + r" \\"


def _lookup(per_rho: Dict[object, Dict[str, object]], rho: float, key: str, scale: float = 1.0):
    """Tra cứu độ đo tại một mức ρ, bỏ qua khác biệt kiểu khoá (float vs str)."""
    for candidate in (rho, f"{rho:g}", str(rho)):
        if candidate in per_rho:
            value = per_rho[candidate].get(key)
            if isinstance(value, dict) or value is None:
                return value
            return float(value) * scale
    return None
