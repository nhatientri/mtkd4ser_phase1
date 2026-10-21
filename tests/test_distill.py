"""Routing + distillation-loss tests.  Run:  pytest -q tests"""

import torch
import torch.nn.functional as F

from mtkd4ser.distill import (
    ConfidenceRouter, CosineRouter, LanguageRouter, UniformRouter, distill_loss, kl_per_sample,
)

EN, FR = 0, 2


def _rand(B=6, K=2, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, generator=g), torch.randn(B, K, C, generator=g)


def test_kl_matches_torch_reference():
    s, t = _rand()
    T = 3.0
    ours = kl_per_sample(s, t, T)                                    # (B,K)
    for k in range(t.shape[1]):
        ref = F.kl_div(F.log_softmax(s / T, -1), F.softmax(t[:, k] / T, -1), reduction="none").sum(-1) * T * T
        assert torch.allclose(ours[:, k], ref, atol=1e-5)


def test_language_routing_is_per_sample():
    """An FR sample must get exactly zero weight on the EN teacher, whatever the batch mix."""
    s, t = _rand(B=4)
    lang = torch.tensor([EN, EN, EN, FR])
    w = LanguageRouter(["en", "fr"])(s, t, lang)
    assert torch.equal(w, torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]]))

    # Loss on the mixed batch == mean of each sample distilled from its own teacher only
    T = 5.0
    loss = distill_loss(s, t, w, T)
    own = torch.stack([kl_per_sample(s[i:i + 1], t[i:i + 1, [0 if l == EN else 1]], T)[0, 0]
                       for i, l in enumerate(lang.tolist())]).mean()
    assert torch.allclose(loss, own, atol=1e-5)


def test_language_routing_differs_from_uniform_on_mixed_batch():
    """Regression for the old bug: batch-averaged weights made language == uniform on a 50/50 batch."""
    s, t = _rand(B=4, seed=1)
    lang = torch.tensor([EN, FR, EN, FR])
    w_lang = LanguageRouter(["en", "fr"])(s, t, lang)
    w_uni = UniformRouter(["en", "fr"])(s, t, lang)
    assert not torch.allclose(distill_loss(s, t, w_lang, 5.0), distill_loss(s, t, w_uni, 5.0))


def test_language_routing_unknown_language_falls_back_to_uniform():
    s, t = _rand(B=2)
    w = LanguageRouter(["en", "fr"])(s, t, torch.tensor([1, EN]))   # 1 = no teacher for it
    assert torch.allclose(w[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(w[1], torch.tensor([1.0, 0.0]))


def test_cosine_routing_varies_per_sample():
    s = torch.tensor([[5., 0, 0, 0], [0, 5., 0, 0]])
    t = torch.tensor([[[5., 0, 0, 0], [0, 5., 0, 0]],
                      [[5., 0, 0, 0], [0, 5., 0, 0]]])
    w = CosineRouter(["en", "fr"], tau=0.25)(s, t, torch.tensor([EN, EN]))
    assert w[0, 0] > 0.9 and w[1, 1] > 0.9
    assert torch.allclose(w.sum(1), torch.ones(2))


def test_confidence_routing_is_not_flat():
    """A confident teacher vs an uncertain one must get clearly different weights."""
    s = torch.zeros(1, 4)
    t = torch.tensor([[[8., 0, 0, 0], [0.2, 0.1, 0, 0]]])
    w = ConfidenceRouter(["en", "fr"], tau=0.1)(s, t, torch.tensor([EN]))
    assert w[0, 0] > 0.95


def test_weights_detached_and_normalised():
    s, t = _rand()
    s.requires_grad_(True)
    lang = torch.tensor([EN, FR] * 3)
    for r in (UniformRouter(["en", "fr"]), LanguageRouter(["en", "fr"]),
              CosineRouter(["en", "fr"]), ConfidenceRouter(["en", "fr"])):
        w = r(s, t, lang)
        assert not w.requires_grad
        assert torch.allclose(w.sum(1), torch.ones(len(s)), atol=1e-6)


def test_single_teacher_kd_is_k1():
    s, t = _rand(K=1)
    w = UniformRouter(["teacher"])(s, t, torch.zeros(len(s), dtype=torch.long))
    assert torch.allclose(distill_loss(s, t, w, 2.0), kl_per_sample(s, t, 2.0).mean())
