"""Collate / attention-mask / normalisation / metrics tests.  Run:  pytest -q tests"""

import numpy as np
import torch

from mtkd4ser.models import SERModel
from mtkd4ser.utils import collate_fn, compute_metrics


def _ex(n, label=0, lang=0, idx=0):
    return {"input_values": np.ones(n, dtype=np.float32), "label": label, "language_id": lang, "idx": idx}


def test_collate_pads_to_batch_max_with_mask():
    b = collate_fn([_ex(3, 1, 0, 7), _ex(5, 2, 2, 8)])
    assert b["input_values"].shape == (2, 5)                 # longest in batch, not 20 s
    assert b["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert b["input_values"][0, 3:].abs().sum() == 0
    assert b["label"].tolist() == [1, 2]
    assert b["language_id"].tolist() == [0, 2]
    assert b["idx"].tolist() == [7, 8]                       # original order kept (needed for teacher cache)


def test_masked_normalisation_ignores_padding():
    x = torch.tensor([[1., 2., 3., 0., 0.], [1., 2., 3., 4., 5.]])
    m = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    y = SERModel._normalize(x, m)
    assert torch.allclose(y[0, :3].mean(), torch.tensor(0.), atol=1e-6)
    assert torch.allclose(y[0, :3].std(unbiased=False), torch.tensor(1.), atol=1e-3)
    assert y[0, 3:].abs().sum() == 0                          # padding stays zero
    # same clip, padded or not, normalises identically
    y_short = SERModel._normalize(x[:1, :3], m[:1, :3])
    assert torch.allclose(y[0, :3], y_short[0], atol=1e-6)


def test_per_language_metrics():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 1, 1, 1, 1]
    langs = [0, 0, 0, 0, 2, 2, 2, 2]
    m = compute_metrics(y_true, y_pred, langs)
    assert m["UR_en"] == 1.0
    assert m["UR_fr"] == 0.25
    assert abs(m["UR_lang_mean"] - 0.625) < 1e-9
    assert m["n_en"] == 4 and m["n_fr"] == 4
