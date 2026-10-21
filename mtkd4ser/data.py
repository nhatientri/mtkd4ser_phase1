"""
data.py – dataset loading for MTKD4SER.

Every loader returns a DatasetDict with train / dev / test splits whose rows have
at least: audio (16 kHz), label (0-3), language_id (0=EN, 2=FR), speaker_id.

Resolution order for each dataset:
  1. local Arrow dataset   data/hf_datasets/<name>/          (export_to_hf_dataset.py)
  2. reformatted CSV splits data/en/iemocap/session<k>/*.csv  (reformat_datasets.py)
                            data/fr/cafe/CaFE_splits/fold<k>/*.csv
  3. Hugging Face Hub       <HF_DATASET_REPO>, config <name>

All three come from reformat_datasets.py, so dev/test are always speaker-independent.
(The old raw-JSON fallback, which made a random – speaker-leaking – dev split, is gone.)
"""

import os
from typing import cast

import numpy as np
import pandas as pd
from datasets import (
    Audio, Dataset, DatasetDict, Features, Value, concatenate_datasets, load_from_disk, load_dataset,
)

from .constants import SAMPLING_RATE

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in mtkd4ser/)
DATA_DIR = os.environ.get("MTKD_DATA_DIR", os.path.join(_PROJECT_DIR, "data"))
HF_DIR = os.path.join(DATA_DIR, "hf_datasets")

SPLITS = ("train", "dev", "test")
KEEP_COLUMNS = ["audio", "label", "language_id", "speaker_id"]


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_local_arrow(name: str) -> DatasetDict | None:
    path = os.path.join(HF_DIR, name)
    if not os.path.isfile(os.path.join(path, "dataset_dict.json")):
        return None
    print(f"[data] {name}: local Arrow dataset {path}")
    return cast(DatasetDict, load_from_disk(path))


def create_hf_split(csv_path, data_root):
    """Load a split CSV (from reformat_datasets.py), resolve full audio paths, and cast to Audio feature."""
    df = pd.read_csv(csv_path)

    # Ensure path exists by prepending data_root if relative
    def resolve_audio(p):
        if os.path.isabs(p) and os.path.isfile(p):
            return p
        candidate1 = os.path.join(data_root, p)
        if os.path.isfile(candidate1):
            return candidate1
        fname = os.path.basename(p)
        for root, _, files in os.walk(data_root):
            if fname in files:
                return os.path.join(root, fname)
        raise FileNotFoundError(f"Audio file not found: {p}")

    df["audio"] = df["audio"].apply(resolve_audio)

    # Convert to standard Python dict
    cols_to_keep = ["audio", "label", "emotion", "language_id", "speaker_id", "gender"]
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    data_dict = {col: df[col].tolist() for col in existing_cols}

    feature_dict = {
        "audio": Audio(sampling_rate=SAMPLING_RATE),
        "label": Value("int64"),
    }
    if "emotion" in existing_cols:
        feature_dict["emotion"] = Value("string")
    if "language_id" in existing_cols:
        feature_dict["language_id"] = Value("int64")
    if "speaker_id" in existing_cols:
        feature_dict["speaker_id"] = Value("string")
    if "gender" in existing_cols:
        feature_dict["gender"] = Value("string")

    return Dataset.from_dict(data_dict, features=Features(feature_dict))


def _load_csv_splits(split_dir: str, name: str) -> DatasetDict | None:
    if not all(os.path.isfile(os.path.join(split_dir, f"{s}.csv")) for s in SPLITS):
        return None
    print(f"[data] {name}: reformatted CSV splits {split_dir}")
    return DatasetDict({s: create_hf_split(os.path.join(split_dir, f"{s}.csv"), DATA_DIR)
                        for s in SPLITS})


def _load_hub(repo_id: str | None, config_name: str) -> DatasetDict | None:
    if not repo_id:
        return None
    print(f"[data] {config_name}: Hugging Face Hub {repo_id}")
    token = os.environ.get("HF_TOKEN")
    return cast(DatasetDict, load_dataset(repo_id, config_name, token=token))


def _select(ds: DatasetDict) -> DatasetDict:
    """Keep only the columns training needs so EN/FR datasets can be concatenated."""
    out = {}
    for split in SPLITS:
        d = ds[split]
        if "speaker_id" not in d.column_names:
            d = d.add_column("speaker_id", ["unknown"] * len(d))
        out[split] = d.select_columns(KEEP_COLUMNS)
    return DatasetDict(out)


def _first(*candidates):
    for fn in candidates:
        ds = fn()
        if ds is not None:
            return ds
    return None


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def iemocap(session: int = 1, hf_repo: str | None = None) -> DatasetDict:
    """IEMOCAP leave-one-session-out fold `session` (1-5)."""
    name = f"iemocap_session{session}"
    ds = _first(
        lambda: _load_local_arrow(name),
        lambda: _load_csv_splits(os.path.join(DATA_DIR, "en", "iemocap", f"session{session}"), name),
        lambda: _load_hub(hf_repo, name),
    )
    if ds is None:
        raise FileNotFoundError(
            f"IEMOCAP session {session} not found locally ({HF_DIR}/{name} or "
            f"{DATA_DIR}/en/iemocap/session{session}/*.csv) and no HF_DATASET_REPO given.")
    return _select(ds)


def cafe(fold: int = 1, hf_repo: str | None = None) -> DatasetDict:
    """CaFE leave-two-speakers-out fold `fold` (1-6). Fold 1 is exported as 'cafe'."""
    arrow_name = "cafe" if fold == 1 else f"cafe_fold{fold}"
    ds = _first(
        lambda: _load_local_arrow(arrow_name),
        lambda: _load_csv_splits(os.path.join(DATA_DIR, "fr", "cafe", "CaFE_splits", f"fold{fold}"), arrow_name),
        lambda: _load_hub(hf_repo, arrow_name),
    )
    if ds is None:
        raise FileNotFoundError(
            f"CaFE fold {fold} not found locally and no HF_DATASET_REPO given "
            f"(Hub config expected: '{arrow_name}'). Run data_prep/export_to_hf_dataset.py --all_cafe_folds "
            "and data_prep/upload_to_hf.py --all_cafe_folds to use folds other than 1.")
    return _select(ds)


def combined(session_en: int = 1, fold_fr: int = 1, hf_repo: str | None = None) -> DatasetDict:
    """EN (IEMOCAP session) + FR (CaFE fold), concatenated per split."""
    ds_en = iemocap(session_en, hf_repo)
    ds_fr = cafe(fold_fr, hf_repo)
    return DatasetDict({
        s: concatenate_datasets([ds_en[s], ds_fr[s]]) for s in SPLITS
    })


def load_task_dataset(linguality: str, language: str, session: int, cafe_fold: int,
                      hf_repo: str | None = None) -> DatasetDict:
    if linguality == "Multilingual":
        return combined(session, cafe_fold, hf_repo)
    if language == "EN":
        return iemocap(session, hf_repo)
    if language == "FR":
        return cafe(cafe_fold, hf_repo)
    raise ValueError(f"Unsupported language: {language}")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(ds: DatasetDict, max_duration: float = 20.0, num_proc: int | None = None) -> DatasetDict:
    """
    Decode audio once into raw float32 waveforms (truncated to max_duration, NOT padded).

    Normalisation (zero-mean / unit-variance) is backbone-specific, so it is done inside
    the model wrapper (models.SERModel). That way the same prepared data can feed a
    wav2vec2 teacher and, e.g., a DistilHuBERT student. Padding happens per batch in
    utils.collate_fn, together with an attention mask.
    """
    max_len = int(SAMPLING_RATE * max_duration)

    def _decode(batch, indices):
        arrays = []
        for a in batch["audio"]:
            if a["sampling_rate"] != SAMPLING_RATE:
                raise ValueError(f"Expected {SAMPLING_RATE} Hz audio, got {a['sampling_rate']}")
            arrays.append(np.asarray(a["array"], dtype=np.float32)[:max_len])
        return {
            "input_values": arrays,
            "length": [len(x) for x in arrays],
            "idx": list(indices),
        }

    out = {}
    for split in ds.keys():
        d = ds[split]
        out[split] = d.map(
            _decode, batched=True, batch_size=64, with_indices=True,
            remove_columns=[c for c in d.column_names if c not in ("label", "language_id", "speaker_id")],
            num_proc=num_proc, desc=f"decode {split}",
        )
    return DatasetDict(out)
