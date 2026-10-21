"""
distill.py – (multi-)teacher distillation loss and routing strategies.

Shapes:  s_logits (B, C)   t_logits (B, K, C)   lang_ids (B,)   weights (B, K)

The KL term is computed PER SAMPLE and weighted PER SAMPLE:

    L_kd = mean_b  sum_k  w[b, k] * T^2 * KL( softmax(t[b,k]/T) || softmax(s[b]/T) )

(The previous version multiplied a batch-averaged KL by a batch-averaged weight, so every
sample was distilled from every teacher in proportion to the batch's language mix –
"language" routing was effectively "uniform" weighted by language frequency.)

Single-teacher KD is the K = 1 case.
"""

import math

import torch
import torch.nn.functional as F

from .constants import LANG_ID_TO_KEY


def kl_per_sample(s_logits, t_logits, temperature):
    """T^2 * KL(teacher || student) for each (sample, teacher). Returns (B, K)."""
    s_log = F.log_softmax(s_logits.float() / temperature, dim=-1).unsqueeze(1)   # (B,1,C)
    t_log = F.log_softmax(t_logits.float() / temperature, dim=-1)                # (B,K,C)
    return (t_log.exp() * (t_log - s_log)).sum(-1) * temperature ** 2


# ---------------------------------------------------------------------------
# Routers – each returns detached weights of shape (B, K) that sum to 1 per row
# ---------------------------------------------------------------------------

class Router:
    name = "base"

    def __init__(self, teacher_langs: list[str]):
        self.teacher_langs = list(teacher_langs)

    def __call__(self, s_logits, t_logits, lang_ids):
        raise NotImplementedError


class UniformRouter(Router):
    name = "uniform"

    def __call__(self, s_logits, t_logits, lang_ids):
        B, K, _ = t_logits.shape
        return torch.full((B, K), 1.0 / K, device=t_logits.device)


class LanguageRouter(Router):
    """Oracle routing: each sample goes to the teacher of its own language (uniform if none)."""
    name = "language"

    def __call__(self, s_logits, t_logits, lang_ids):
        B, K, _ = t_logits.shape
        w = torch.full((B, K), 1.0 / K, device=t_logits.device)
        for col, key in enumerate(self.teacher_langs):
            for lid, lkey in LANG_ID_TO_KEY.items():
                if lkey == key:
                    sel = lang_ids == lid
                    if sel.any():
                        w[sel] = 0.0
                        w[sel, col] = 1.0
        return w


class CosineRouter(Router):
    """Per-sample cosine similarity between student and teacher logits, softmax over teachers / tau."""
    name = "cosine"

    def __init__(self, teacher_langs, tau=0.25):
        super().__init__(teacher_langs)
        self.tau = tau

    def __call__(self, s_logits, t_logits, lang_ids):
        sim = F.cosine_similarity(s_logits.float().unsqueeze(1), t_logits.float(), dim=-1)  # (B,K)
        return F.softmax(sim / self.tau, dim=-1).detach()


class ConfidenceRouter(Router):
    """
    Per-sample teacher confidence = 1 - normalised entropy of the teacher's T=1 prediction
    (in [0, 1]), sharpened with softmax(conf / tau). The old version used T=5-softened
    max-probabilities with tau=1, which gave ~0.51/0.49 weights – effectively uniform.
    """
    name = "confidence"

    def __init__(self, teacher_langs, tau=0.1):
        super().__init__(teacher_langs)
        self.tau = tau

    def __call__(self, s_logits, t_logits, lang_ids):
        p = F.softmax(t_logits.float(), dim=-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)                  # (B,K)
        conf = 1.0 - ent / math.log(t_logits.shape[-1])
        return F.softmax(conf / self.tau, dim=-1).detach()


def make_router(mode: str, teacher_langs: list[str], cosine_tau=0.25, conf_tau=0.1) -> Router:
    if mode == "uniform":
        return UniformRouter(teacher_langs)
    if mode == "language":
        return LanguageRouter(teacher_langs)
    if mode == "cosine":
        return CosineRouter(teacher_langs, cosine_tau)
    if mode == "confidence":
        return ConfidenceRouter(teacher_langs, conf_tau)
    raise ValueError(f"Unknown routing mode: {mode}")


def distill_loss(s_logits, t_logits, weights, temperature):
    """Per-sample weighted KD loss (scalar)."""
    kl = kl_per_sample(s_logits, t_logits, temperature)    # (B,K)
    return (weights * kl).sum(dim=1).mean()
