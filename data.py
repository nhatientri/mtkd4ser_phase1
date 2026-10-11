"""
data.py – Phase 1
- Adds language_id to every sample (0=EN, 1=FI, 2=FR)
- Proper dev split for IEMOCAP & CaFE (no longer dev=test)
- Colab-friendly default roots
- Works even when FESC is unavailable (EN+FR only)
"""

import os
import json
import glob
from typing import cast
import pandas as pd
from datasets import Dataset, DatasetDict, Audio, concatenate_datasets, load_from_disk, load_dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from utils import (
    update_iemocap_label,
    update_fesc_label,
    is_common,
    to_label,
)


# ---------------------------------------------------------------------------
# Default roots – auto-detects local preprocessed data, HF Hub repo, or Colab defaults
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data")

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", None)

IEMOCAP_ROOT = os.environ.get(
    "IEMOCAP_ROOT",
    os.path.join(_DATA_DIR, "en", "iemocap") if os.path.isdir(os.path.join(_DATA_DIR, "en", "iemocap")) else "/content/IEMOCAP"
)
CAFE_ROOT = os.environ.get(
    "CAFE_ROOT",
    os.path.join(_DATA_DIR, "fr", "cafe") if os.path.isdir(os.path.join(_DATA_DIR, "fr", "cafe")) else "/content/CaFE"
)
FESC_ROOT = os.environ.get(
    "FESC_ROOT",
    os.path.join(_DATA_DIR, "fi", "fesc") if os.path.isdir(os.path.join(_DATA_DIR, "fi", "fesc")) else "/content/FESC"
)


LABELS = ["anger", "happiness", "neutral", "sadness"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def _make_label_maps():
    return LABEL2ID.copy(), ID2LABEL.copy()


def _get_hf_repo(hf_repo_arg=None):
    return hf_repo_arg or os.environ.get("HF_DATASET_REPO", None) or HF_DATASET_REPO


def _load_local_hf_dir(path: str) -> DatasetDict | None:
    if not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "dataset_info.json")):
        return cast(DatasetDict, load_from_disk(path))
    try:
        return cast(DatasetDict, load_dataset(path))
    except Exception:
        try:
            return cast(DatasetDict, load_from_disk(path))
        except Exception:
            return None


def _load_hf_hub_dataset(repo_id: str, config_name: str) -> DatasetDict:
    token = os.environ.get("HF_TOKEN", None)
    try:
        return cast(DatasetDict, load_dataset(repo_id, config_name, token=token))
    except TypeError:
        return cast(DatasetDict, load_dataset(repo_id, config_name, use_auth_token=token))


# ---------------------------------------------------------------------------
# IEMOCAP
# ---------------------------------------------------------------------------

def iemocap(
    session=1, root=None, create_dev=True, dev_ratio=0.1, seed=42, hf_dataset_repo=None
) -> tuple[dict[str, int], dict[int, str], DatasetDict]:
    """
    Load one IEMOCAP session (leave-one-session-out style).
    Creates a real dev split from the training speakers (no leakage).
    Supports direct Arrow zero-copy loading if available.
    """
    label2id, id2label = _make_label_maps()
    root = root or IEMOCAP_ROOT

    # Check for pre-built Hugging Face Arrow dataset first (fastest)
    hf_candidates = [
        os.path.join(_DATA_DIR, "hf_datasets", f"iemocap_session{session}"),
        os.path.join(root, "hf_datasets", f"iemocap_session{session}"),
        os.path.join(os.path.dirname(root), "hf_datasets", f"iemocap_session{session}"),
    ]
    for hf_p in hf_candidates:
        ds_local = _load_local_hf_dir(hf_p)
        if ds_local is not None:
            print(f"[INFO] Loading IEMOCAP session {session} from local path: {hf_p}")
            return label2id, id2label, ds_local

    hf_repo = _get_hf_repo(hf_dataset_repo)
    if hf_repo:
        print(f"[INFO] Loading IEMOCAP session {session} from HF Hub: {hf_repo}")
        return label2id, id2label, _load_hf_hub_dataset(hf_repo, f"iemocap_session{session}")
    # Expect pre-built json splits (same format as original paper)
    # Fallback: try common Kaggle layout
    train_json = f"{root}/session{session}/train.json"
    test_json  = f"{root}/session{session}/test.json"

    if not os.path.isfile(train_json):
        # Alternative layout used by some Kaggle uploads
        train_json = f"{root}/IEMOCAP_splits/session{session}/train.json"
        test_json  = f"{root}/IEMOCAP_splits/session{session}/test.json"

    if not os.path.isfile(train_json):
        raise FileNotFoundError(
            f"IEMOCAP train split not found for session {session}.\n"
            f"Looked at: {train_json}\n"
            "Prepare the json splits or set IEMOCAP_ROOT correctly."
        )

    with open(train_json) as f:
        train_data = json.load(f)
    with open(test_json) as f:
        test_data = json.load(f)

    train_df = pd.DataFrame.from_dict(train_data, orient="index").reset_index()
    test_df  = pd.DataFrame.from_dict(test_data,  orient="index").reset_index()

    train_df = train_df.rename(columns={"index": "file_id", "wav": "audio"})
    test_df  = test_df.rename(columns={"index": "file_id", "wav": "audio"})

    # Keep only the 4 common emotions (+ map excited → happiness)
    def _filter_and_map(df):
        df = df.copy()
        df["emo"] = df["emo"].apply(update_iemocap_label)
        df = df[df["emo"].isin(LABELS)].reset_index(drop=True)
        return df

    train_df = _filter_and_map(train_df)
    test_df  = _filter_and_map(test_df)

    label2id, id2label = _make_label_maps()
    train_df["label"] = train_df["emo"].apply(lambda e: to_label(e, label2id))
    test_df["label"]  = test_df["emo"].apply(lambda e: to_label(e, label2id))
    train_df["language_id"] = 0
    test_df["language_id"]  = 0

    # Real dev split from train (stratified)
    if create_dev and len(train_df) > 20:
        tr_idx, dv_idx = train_test_split(
            train_df.index,
            test_size=dev_ratio,
            stratify=train_df["label"],
            random_state=seed,
        )
        dev_df   = train_df.loc[dv_idx].reset_index(drop=True)
        train_df = train_df.loc[tr_idx].reset_index(drop=True)
    else:
        dev_df = test_df.copy()  # last resort

    def _to_ds(df):
        ds = Dataset.from_pandas(df[["audio", "label", "language_id"]])
        return ds.cast_column("audio", Audio(sampling_rate=16_000))

    ds = DatasetDict({
        "train": _to_ds(train_df),
        "dev":   _to_ds(dev_df),
        "test":  _to_ds(test_df),
    })
    return label2id, id2label, ds


# ---------------------------------------------------------------------------
# CaFE
# ---------------------------------------------------------------------------

def cafe(
    session=1, root=None, create_dev=True, dev_ratio=0.15, seed=42, hf_dataset_repo=None
) -> tuple[dict[str, int], dict[int, str], DatasetDict]:
    """
    Load CaFE (Canadian French).
    Supports direct Arrow zero-copy loading if available.
    Expects either the original paper's json splits or raw wavs + metadata.
    """
    label2id, id2label = _make_label_maps()
    root = root or CAFE_ROOT

    # Check for pre-built Hugging Face Arrow dataset first (fastest)
    hf_candidates = [
        os.path.join(_DATA_DIR, "hf_datasets", "cafe"),
        os.path.join(root, "hf_datasets", "cafe"),
        os.path.join(os.path.dirname(root), "hf_datasets", "cafe"),
    ]
    for hf_p in hf_candidates:
        ds_local = _load_local_hf_dir(hf_p)
        if ds_local is not None:
            print(f"[INFO] Loading CaFE from local path: {hf_p}")
            return label2id, id2label, ds_local

    hf_repo = _get_hf_repo(hf_dataset_repo)
    if hf_repo:
        print(f"[INFO] Loading CaFE from HF Hub: {hf_repo}")
        return label2id, id2label, _load_hf_hub_dataset(hf_repo, "cafe")

    splits_dir = os.path.join(root, "CaFE_splits")
    if not os.path.isdir(splits_dir):
        splits_dir = root  # maybe jsons are directly under root

    all_files = glob.glob(os.path.join(splits_dir, "*.json"))
    if not all_files:
        raise FileNotFoundError(
            f"No CaFE json splits found under {splits_dir}.\n"
            "Download CaFE from Zenodo and prepare splits, "
            "or place the paper's CaFE_splits folder there."
        )

    rows = []
    for fp in all_files:
        with open(fp) as f:
            data = json.load(f)
        split_name = os.path.basename(fp).split(".")[0]
        for k, v in data.items():
            rows.append({
                "audio": v.get("wav", v.get("audio", k)),
                "label": v.get("emo", v.get("label")),
                "set": split_name,
            })

    df = pd.DataFrame(rows)
    df["label_flag"] = df["label"].apply(is_common)
    df = df[df["label_flag"] == 1].reset_index(drop=True)

    # Fix paths
    def _fix_path(p):
        if os.path.isfile(p):
            return p
        fname = os.path.basename(p)
        # search under root
        hits = glob.glob(os.path.join(root, "**", fname), recursive=True)
        return hits[0] if hits else p

    df["audio"] = df["audio"].apply(_fix_path)

    label2id, id2label = _make_label_maps()
    df["label"] = df["label"].apply(lambda e: to_label(e, label2id))
    df["language_id"] = 2  # French

    train_df = df[df["set"] != "test"].reset_index(drop=True)
    test_df  = df[df["set"] == "test"].reset_index(drop=True)

    if create_dev and len(train_df) > 15:
        tr_idx, dv_idx = train_test_split(
            train_df.index,
            test_size=dev_ratio,
            stratify=train_df["label"],
            random_state=seed,
        )
        dev_df   = train_df.loc[dv_idx].reset_index(drop=True)
        train_df = train_df.loc[tr_idx].reset_index(drop=True)
    else:
        dev_df = test_df.copy()

    def _to_ds(d):
        ds = Dataset.from_pandas(d[["audio", "label", "language_id"]])
        return ds.cast_column("audio", Audio(sampling_rate=16_000))

    ds = DatasetDict({
        "train": _to_ds(train_df),
        "dev":   _to_ds(dev_df),
        "test":  _to_ds(test_df),
    })
    return label2id, id2label, ds


# ---------------------------------------------------------------------------
# FESC (Finnish) – optional
# ---------------------------------------------------------------------------

def fesc(session=9, root=None) -> tuple[dict[str, int], dict[int, str], DatasetDict]:
    """
    Load FESC if available. Raises a clear error otherwise so the user
    can fall back to EN+FR only.
    """
    root = root or FESC_ROOT
    SESSION_FOLDER_MAP = {
        1: "TIRE", 2: "TIPE", 3: "JARA", 4: "MIKO", 5: "ANRO",
        6: "RIGR", 7: "TUVA", 8: "PEKO", 9: "JAKA",
    }
    folder = SESSION_FOLDER_MAP.get(session, "ERROR")
    train_json = f"{root}/{folder}/train.json"
    test_json  = f"{root}/{folder}/test.json"
    dev_json   = f"{root}/{folder}/dev.json"

    for p in (train_json, test_json, dev_json):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"FESC split not found: {p}\n"
                "FESC is not publicly redistributable. "
                "Either obtain it from the authors or run EN+FR only."
            )

    def _load(path):
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df = df.rename(columns={"index": "file_id", "label": "emo", "file_path": "audio"})
        df = df[df["emo"] != "5"].reset_index(drop=True)
        df["emo"] = df["emo"].apply(update_fesc_label)
        return df

    train_df = _load(train_json)
    test_df  = _load(test_json)
    dev_df   = _load(dev_json)

    label2id, id2label = _make_label_maps()
    for df in (train_df, test_df, dev_df):
        df["label"] = df["emo"].apply(lambda e: to_label(e, label2id))
        df["language_id"] = 1  # Finnish

    def _to_ds(d):
        ds = Dataset.from_pandas(d[["audio", "label", "language_id"]])
        return ds.cast_column("audio", Audio(sampling_rate=16_000))

    ds = DatasetDict({
        "train": _to_ds(train_df),
        "dev":   _to_ds(dev_df),
        "test":  _to_ds(test_df),
    })
    return label2id, id2label, ds


# ---------------------------------------------------------------------------
# Multilingual concatenation helper
# ---------------------------------------------------------------------------

def build_multilingual(
    session_en=1, session_fi=9, session_fr=1, include_fi=False, hf_dataset_repo=None
) -> tuple[dict[str, int], dict[int, str], DatasetDict]:
    """
    Concatenate EN (+ optional FI) + FR.
    Uses pre-built combined HF Arrow dataset if available for Session 1.
    """
    label2id, id2label = _make_label_maps()
    hf_repo = _get_hf_repo(hf_dataset_repo)
    hf_comb = os.path.join(_DATA_DIR, "hf_datasets", "combined")
    if session_en == 1 and session_fr == 1 and not include_fi:
        ds_local = _load_local_hf_dir(hf_comb)
        if ds_local is not None:
            print(f"[INFO] Loading Combined Multilingual (EN Session 1 + FR Fold 1) from local path: {hf_comb}")
            return label2id, id2label, ds_local
        elif hf_repo:
            print(f"[INFO] Loading Combined Multilingual from HF Hub: {hf_repo}")
            return label2id, id2label, _load_hf_hub_dataset(hf_repo, "combined")

    label2id, id2label, ds_en = iemocap(session=session_en, hf_dataset_repo=hf_repo)
    _, _, ds_fr = cafe(session=session_fr, hf_dataset_repo=hf_repo)

    datasets: list[DatasetDict] = [ds_en, ds_fr]
    if include_fi:
        try:
            _, _, ds_fi = fesc(session=session_fi)
            datasets.append(ds_fi)
        except FileNotFoundError as e:
            print(f"[WARN] FESC unavailable – continuing with EN+FR only.\n{e}")

    train = concatenate_datasets([cast(Dataset, d["train"]) for d in datasets])
    dev   = concatenate_datasets([cast(Dataset, d["dev"])   for d in datasets])
    test  = concatenate_datasets([cast(Dataset, d["test"])  for d in datasets])

    return label2id, id2label, DatasetDict({
        "train": train, "dev": dev, "test": test
    })
