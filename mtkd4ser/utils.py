"""
utils.py – batching and metrics.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score

from .constants import LANG_ID_TO_KEY, NUM_CLASSES


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Pad raw waveforms to the longest clip in *this batch* and build an attention mask
    (1 = real audio, 0 = padding). The mask is passed to the model so that padding is
    ignored by self-attention and by the mean-pooling in the classification head.
    """
    lengths = torch.tensor([len(ex["input_values"]) for ex in batch], dtype=torch.long)
    max_len = int(lengths.max())
    inputs = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, ex in enumerate(batch):
        x = torch.as_tensor(np.asarray(ex["input_values"], dtype=np.float32))
        inputs[i, : len(x)] = x
    attention_mask = (torch.arange(max_len)[None, :] < lengths[:, None]).long()

    out = {
        "input_values": inputs,
        "attention_mask": attention_mask,
        "label": torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long),
        "language_id": torch.tensor([int(ex.get("language_id", 0)) for ex in batch], dtype=torch.long),
    }
    if "idx" in batch[0]:
        out["idx"] = torch.tensor([int(ex["idx"]) for ex in batch], dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, lang_ids=None) -> dict:
    """
    UR = unweighted (macro) recall, WR = weighted recall, acc.
    If lang_ids is given, also UR/WR/acc per language (UR_en, UR_fr, ...) and
    UR_lang_mean = mean of per-language UR (the model-selection metric for
    multilingual runs, so the small FR set is not drowned out by EN).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(NUM_CLASSES))

    def _block(t, p):
        if len(t) == 0:
            return {"UR": float("nan"), "WR": float("nan"), "acc": float("nan"), "n": 0}
        present = sorted(set(t.tolist()))
        return {
            "UR": float(recall_score(t, p, labels=present, average="macro", zero_division=0)),
            "WR": float(recall_score(t, p, labels=labels, average="weighted", zero_division=0)),
            "acc": float((t == p).mean()),
            "n": int(len(t)),
        }

    m = _block(y_true, y_pred)
    out = {"UR": m["UR"], "WR": m["WR"], "acc": m["acc"], "n": m["n"]}

    if lang_ids is not None:
        lang_ids = np.asarray(lang_ids)
        per_lang_ur = []
        for lid in sorted(set(lang_ids.tolist())):
            key = LANG_ID_TO_KEY.get(lid, str(lid))
            sel = lang_ids == lid
            b = _block(y_true[sel], y_pred[sel])
            out[f"UR_{key}"] = b["UR"]
            out[f"WR_{key}"] = b["WR"]
            out[f"acc_{key}"] = b["acc"]
            out[f"n_{key}"] = b["n"]
            per_lang_ur.append(b["UR"])
        out["UR_lang_mean"] = float(np.mean(per_lang_ur)) if per_lang_ur else out["UR"]
    return out


def print_confusion(y_true, y_pred, label_names=None, title="Confusion matrix"):
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = label_names or [str(i) for i in labels]
    w = max(len(n) for n in names) + 2
    print(f"{title} (rows = true, cols = predicted)")
    print(" " * w + "".join(f"{n[:7]:>9}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n:<{w}}" + "".join(f"{v:>9d}" for v in row))
    return cm
