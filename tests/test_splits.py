"""No speaker may appear in more than one of train/dev/test. Skipped if the data isn't present."""

import os

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FOLDS = [os.path.join(ROOT, "en", "iemocap", f"session{k}") for k in range(1, 6)] + \
        [os.path.join(ROOT, "fr", "cafe", "CaFE_splits", f"fold{k}") for k in range(1, 7)] + \
        [os.path.join(ROOT, "combined")]


@pytest.mark.parametrize("fold_dir", FOLDS, ids=lambda p: os.path.relpath(p, ROOT))
def test_no_speaker_overlap(fold_dir):
    paths = {s: os.path.join(fold_dir, f"{s}.csv") for s in ("train", "dev", "test")}
    if not all(os.path.isfile(p) for p in paths.values()):
        pytest.skip(f"splits not found in {fold_dir}")
    spk = {s: set(pd.read_csv(p)["speaker_id"]) for s, p in paths.items()}
    assert not spk["train"] & spk["dev"]
    assert not spk["train"] & spk["test"]
    assert not spk["dev"] & spk["test"]
