"""Khung trọng số--đích thống nhất (Mục 4.1 của bản thảo).

Mọi phương pháp trong bài đều viết được dưới dạng

    L = (1/B) Σ CE(z_i, y_i) + λ (1/B) Σ w_i · D(t_i ‖ q_i),

với z = f_θ(x) trên ảnh gốc, q = softmax(f_θ(x̃)) trên ảnh bị che, còn w_i ≥ 0 và
t_i ∈ Δ^{C-1} không nhận gradient. Mỗi phương pháp chỉ khác nhau ở cặp (w_i, t_i):

    E3  consistency cố định         w = 1,            t = p
    E4  visibility-gating           w = g(v),         t = p
    E5  visibility + confidence     w = g(v)·h(c),    t = p
    SA  Soft Augmentation           w = 1,            t = α(v)·e_y + (1-α(v))·u
    E6  VCC (đề xuất)               w = h(c),         t = α(v)·p  + (1-α(v))·u

trong đó α(v) = 1 - (1-v)^k.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Hàm cổng
# --------------------------------------------------------------------------- #
def alpha_from_visibility(visibility: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    """α(v) = 1 - (1-v)^k, không giảm theo v, α(0)=0, α(1)=1."""
    v = visibility.clamp(0.0, 1.0)
    return 1.0 - (1.0 - v).pow(k)


def visibility_gate(visibility: torch.Tensor, cfg) -> torch.Tensor:
    """g(v) ∈ [0,1], không giảm theo v (Bảng 2 của bản thảo)."""
    kind = str(cfg.get("kind", "none")).lower()
    v = visibility.clamp(0.0, 1.0)
    if kind in ("none", "const", "one"):
        return torch.ones_like(v)
    if kind == "linear":
        return v
    if kind == "poly":
        return alpha_from_visibility(v, float(cfg.get("k", 2.0)))
    if kind == "hard":
        return (v >= float(cfg.get("v0", 0.5))).to(v.dtype)
    if kind == "sigmoid":
        v0 = float(cfg.get("v0", 0.5))
        tau = max(float(cfg.get("tau", 0.1)), EPS)
        return torch.sigmoid((v - v0) / tau)
    raise ValueError(f"Hàm cổng visibility không hỗ trợ: {kind}")


def confidence_gate(confidence: torch.Tensor, cfg) -> torch.Tensor:
    """h(c) ∈ [0,1]: lọc theo độ tin cậy của dự đoán trên ảnh gốc."""
    kind = str(cfg.get("kind", "none")).lower()
    c = confidence.clamp(0.0, 1.0)
    if kind in ("none", "const", "one"):
        return torch.ones_like(c)
    if kind in ("linear", "prob"):
        return c
    if kind == "hard":  # FixMatch
        return (c >= float(cfg.get("tau", 0.8))).to(c.dtype)
    if kind == "softmatch":  # Gaussian cắt cụt (Chen và cộng sự, 2023)
        mu = float(cfg.get("mu", 0.7))
        sigma = max(float(cfg.get("sigma", 0.2)), EPS)
        soft = torch.exp(-((c - mu) ** 2) / (2 * sigma**2))
        return torch.where(c >= mu, torch.ones_like(c), soft)
    if kind == "power":
        return c.pow(float(cfg.get("power", 2.0)))
    raise ValueError(f"Hàm cổng confidence không hỗ trợ: {kind}")


def confidence_value(probs: torch.Tensor, labels: torch.Tensor, source: str = "true_class") -> torch.Tensor:
    """c = p_y (mặc định, theo bản thảo) hoặc c = max_k p_k."""
    source = str(source).lower()
    if source in ("true_class", "p_y", "label"):
        return probs.gather(1, labels.view(-1, 1)).squeeze(1)
    if source in ("max", "max_prob"):
        return probs.max(dim=1).values
    raise ValueError(f"Nguồn độ tin cậy không hỗ trợ: {source}")


# --------------------------------------------------------------------------- #
# Phân phối đích
# --------------------------------------------------------------------------- #
def build_target(
    probs_clean: torch.Tensor,
    logits_clean: torch.Tensor,
    labels: torch.Tensor,
    visibility: torch.Tensor,
    cfg,
    prior: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trả về (đích t, α(v)). ``t`` luôn đã bị chặn gradient."""
    kind = str(cfg.get("kind", "pred")).lower()
    num_classes = probs_clean.shape[1]
    k = float(cfg.get("k", 2.0))
    alpha = alpha_from_visibility(visibility, k).view(-1, 1)

    if prior is None:
        base = torch.full_like(probs_clean, 1.0 / num_classes)
    else:
        base = prior.to(probs_clean.device, probs_clean.dtype).view(1, -1).expand_as(probs_clean)

    if kind in ("pred", "p", "consistency"):
        target = probs_clean
        alpha = torch.ones_like(alpha)
    elif kind in ("vcc", "mix"):
        target = alpha * probs_clean + (1.0 - alpha) * base
    elif kind in ("soft_aug", "sa"):
        onehot = F.one_hot(labels, num_classes).to(probs_clean.dtype)
        target = alpha * onehot + (1.0 - alpha) * base
    elif kind in ("vcc_temp", "temperature"):
        # Biến thể E6-T: làm mềm bằng nhiệt độ thay vì trộn với phân phối đều.
        t_max = float(cfg.get("t_max", 4.0))
        temperature = 1.0 + (t_max - 1.0) * (1.0 - alpha)
        target = F.softmax(logits_clean / temperature.clamp(min=EPS), dim=1)
    elif kind in ("onehot", "label"):
        target = F.one_hot(labels, num_classes).to(probs_clean.dtype)
        alpha = torch.ones_like(alpha)
    else:
        raise ValueError(f"Kiểu đích không hỗ trợ: {kind}")

    return target.detach(), alpha.detach().squeeze(1)


def soften_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """p = softmax(z / T); T > 1 dùng để hiệu chỉnh mô hình thầy trước khi trộn."""
    if abs(temperature - 1.0) < 1e-6:
        return F.softmax(logits, dim=1)
    return F.softmax(logits / max(temperature, EPS), dim=1)


# --------------------------------------------------------------------------- #
# Hàm phân kỳ
# --------------------------------------------------------------------------- #
def divergence(target: torch.Tensor, logits_occ: torch.Tensor, kind: str = "kl") -> torch.Tensor:
    """D(t ‖ q) cho từng mẫu; ``q = softmax(logits_occ)``."""
    kind = str(kind).lower()
    log_q = F.log_softmax(logits_occ, dim=1)
    if kind == "kl":
        log_t = target.clamp_min(EPS).log()
        return (target * (log_t - log_q)).sum(dim=1)
    if kind == "ce":  # cross-entropy với nhãn mềm = KL + H(t) (hằng số theo θ)
        return -(target * log_q).sum(dim=1)
    if kind == "js":
        q = log_q.exp()
        mixture = (0.5 * (target + q)).clamp_min(EPS)
        log_m = mixture.log()
        kl_t = (target * (target.clamp_min(EPS).log() - log_m)).sum(dim=1)
        kl_q = (q * (log_q - log_m)).sum(dim=1)
        return 0.5 * (kl_t + kl_q)
    if kind == "mse":
        q = log_q.exp()
        return (q - target).pow(2).sum(dim=1) / target.shape[1]
    raise ValueError(f"Hàm phân kỳ không hỗ trợ: {kind}")


# --------------------------------------------------------------------------- #
# Hàm mục tiêu hợp nhất
# --------------------------------------------------------------------------- #
class UnifiedConsistencyLoss(nn.Module):
    """Cài đặt phương trình (4)/(5): CE + λ·w·D(t‖q).

    Cấu hình (nhánh ``loss`` trong YAML)::

        ce_branch: clean            # clean | occ | both
        label_smoothing: 0.0
        consistency:
          enabled: true
          lambda: 1.0
          divergence: kl           # kl | ce | js | mse
          reduction: mean          # mean | weighted (Σ wD / Σ w)
          target:
            kind: vcc              # pred | vcc | soft_aug | vcc_temp | onehot
            k: 2.0
            prior: uniform         # uniform | class_freq
            teacher_temperature: 1.0
          gate:
            combine: product       # product | min
            visibility: {kind: none}
            confidence: {kind: linear, source: true_class}
    """

    def __init__(self, cfg, class_prior: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.ce_branch = str(cfg.get("ce_branch", "clean")).lower()
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))

        cons = cfg.get("consistency", {})
        self.enabled = bool(cons.get("enabled", False))
        self.lambda_cons = float(cons.get("lambda", 1.0))
        self.divergence_kind = str(cons.get("divergence", "kl"))
        self.reduction = str(cons.get("reduction", "mean")).lower()
        self.target_cfg = cons.get("target", {})
        self.gate_cfg = cons.get("gate", {})
        self.combine = str(self.gate_cfg.get("combine", "product")).lower()
        self.teacher_temperature = float(self.target_cfg.get("teacher_temperature", 1.0))

        prior_kind = str(self.target_cfg.get("prior", "uniform")).lower()
        if prior_kind == "class_freq" and class_prior is not None:
            self.register_buffer("prior", class_prior.clone())
        else:
            self.register_buffer("prior", None)

    # ------------------------------------------------------------------ #
    def weights(self, probs_clean: torch.Tensor, labels: torch.Tensor, visibility: torch.Tensor):
        gate_v = visibility_gate(visibility, self.gate_cfg.get("visibility", {}))
        conf_cfg = self.gate_cfg.get("confidence", {})
        confidence = confidence_value(probs_clean, labels, conf_cfg.get("source", "true_class"))
        gate_c = confidence_gate(confidence, conf_cfg)
        if self.combine == "min":
            weight = torch.minimum(gate_v, gate_c)
        elif self.combine == "product":
            weight = gate_v * gate_c
        else:
            raise ValueError(f"Cách kết hợp cổng không hỗ trợ: {self.combine}")
        return weight.detach(), gate_v.detach(), gate_c.detach(), confidence.detach()

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_occ: Optional[torch.Tensor],
        labels: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        lambda_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}
        if logits_clean is None and (self.enabled or self.ce_branch in ("clean", "both")):
            raise ValueError("Cấu hình này cần logits của nhánh ảnh gốc (logits_clean).")

        # --- hạng cross-entropy có giám sát ---
        ce_terms = []
        if self.ce_branch in ("clean", "both"):
            ce_terms.append(F.cross_entropy(logits_clean, labels, label_smoothing=self.label_smoothing))
        if self.ce_branch in ("occ", "both"):
            if logits_occ is None:
                raise ValueError("ce_branch yêu cầu nhánh ảnh bị che nhưng logits_occ=None")
            ce_terms.append(F.cross_entropy(logits_occ, labels, label_smoothing=self.label_smoothing))
        if not ce_terms:
            raise ValueError(f"ce_branch không hợp lệ: {self.ce_branch}")
        ce_loss = torch.stack(ce_terms).mean()
        loss = ce_loss
        stats["ce"] = float(ce_loss.detach())

        # --- hạng consistency ---
        if self.enabled and logits_occ is not None:
            if visibility is None:
                visibility = torch.ones(logits_clean.shape[0], device=logits_clean.device)
            probs_clean = soften_probs(logits_clean, self.teacher_temperature).detach()
            target, alpha = build_target(
                probs_clean, logits_clean.detach(), labels, visibility, self.target_cfg, self.prior
            )
            weight, gate_v, gate_c, confidence = self.weights(probs_clean, labels, visibility)
            per_sample = divergence(target, logits_occ, self.divergence_kind)

            if self.reduction == "weighted":
                cons_loss = (weight * per_sample).sum() / weight.sum().clamp(min=EPS)
            elif self.reduction == "mean":
                cons_loss = (weight * per_sample).mean()
            else:
                raise ValueError(f"Cách thu gọn không hỗ trợ: {self.reduction}")

            lambda_eff = self.lambda_cons * float(lambda_scale)
            loss = loss + lambda_eff * cons_loss

            with torch.no_grad():
                stats.update(
                    cons=float(cons_loss.detach()),
                    cons_raw=float(per_sample.mean()),
                    lambda_eff=lambda_eff,
                    w_mean=float(weight.mean()),
                    gate_v_mean=float(gate_v.mean()),
                    gate_c_mean=float(gate_c.mean()),
                    conf_clean=float(confidence.mean()),
                    alpha_mean=float(alpha.mean()),
                    target_max=float(target.max(dim=1).values.mean()),
                    target_entropy=float(-(target.clamp_min(EPS) * target.clamp_min(EPS).log()).sum(1).mean()),
                    q_conf=float(F.softmax(logits_occ.detach(), dim=1).max(dim=1).values.mean()),
                    v_mean=float(visibility.mean()),
                )
        stats["loss"] = float(loss.detach())
        return loss, stats


def build_criterion(cfg, class_prior: Optional[torch.Tensor] = None) -> UnifiedConsistencyLoss:
    return UnifiedConsistencyLoss(cfg.loss, class_prior=class_prior)


def needs_occluded_branch(cfg) -> bool:
    """Cấu hình có cần nhánh ảnh bị che hay không (E0 thì không)."""
    loss_cfg = cfg.loss
    if str(loss_cfg.get("ce_branch", "clean")).lower() in ("occ", "both"):
        return True
    return bool(loss_cfg.get("consistency", {}).get("enabled", False))
