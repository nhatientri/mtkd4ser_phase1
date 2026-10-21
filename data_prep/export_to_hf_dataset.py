#!/usr/bin/env python3
"""
export_to_hf_dataset.py
-----------------------
Exports the preformatted SER datasets into Hugging Face Dataset format
(Apache Arrow via save_to_disk) for ultra-fast, zero-copy training on Colab/GPU.

Produces:
1. data/hf_datasets/combined/
   - train, dev, test splits with Audio(sampling_rate=16_000), label, language_id, speaker_id, gender
2. data/hf_datasets/iemocap_session1/ ... iemocap_session5/
3. data/hf_datasets/cafe/

Also supports exporting single .parquet files alongside for archival/hub upload.
"""

import os
import sys
import argparse
from datasets import DatasetDict, load_from_disk

# Run as `python data_prep/export_to_hf_dataset.py`: make the repo root importable for `mtkd4ser`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtkd4ser.data import create_hf_split  # noqa: E402


def export_dataset_dict(split_paths, data_root, output_dir, export_parquet=False):
    """Build and save a DatasetDict to disk as Arrow."""
    os.makedirs(output_dir, exist_ok=True)
    splits = {}
    for split_name, csv_p in split_paths.items():
        if os.path.isfile(csv_p):
            print(f"  Processing split '{split_name}' from {csv_p}...")
            splits[split_name] = create_hf_split(csv_p, data_root)

    dataset_dict = DatasetDict(splits)

    print(f"  Saving Arrow dataset to: {output_dir}...")
    dataset_dict.save_to_disk(output_dir)

    if export_parquet:
        parquet_dir = os.path.join(output_dir, "parquet")
        os.makedirs(parquet_dir, exist_ok=True)
        for s_name, s_ds in dataset_dict.items():
            out_p = os.path.join(parquet_dir, f"{s_name}.parquet")
            print(f"  Exporting Parquet: {out_p}...")
            s_ds.to_parquet(out_p)

    return dataset_dict


def verify_hf_dataset(hf_dir):
    """Verify loading from disk and sampling audio."""
    print(f"\nVerifying {hf_dir}...")
    ds = load_from_disk(hf_dir)
    print("  Structure:", ds)
    for split in ds.keys():
        sample = ds[split][0]
        audio_info = sample["audio"]
        print(f"  [{split}] Sample 0:")
        print(f"    emotion: {sample.get('emotion')}, label: {sample.get('label')}, lang_id: {sample.get('language_id')}")
        print(f"    speaker: {sample.get('speaker_id')}, gender: {sample.get('gender')}")
        print(f"    audio sample rate: {audio_info['sampling_rate']}, array len: {len(audio_info['array'])}")
        assert audio_info["sampling_rate"] == 16000, f"Expected 16kHz, got {audio_info['sampling_rate']}"
        assert len(audio_info["array"]) > 0, "Empty audio waveform"
    print("  Verification PASSED!")


def main():
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    _default_data_dir = os.path.join(_project_dir, "data")
    _default_out_dir = os.path.join(_default_data_dir, "hf_datasets")

    parser = argparse.ArgumentParser(description="Export SER datasets to Hugging Face Arrow format.")
    parser.add_argument("--data_dir", default=_default_data_dir,
                        help="Path to preformatted data directory")
    parser.add_argument("--out_dir", default=_default_out_dir,
                        help="Path to output Hugging Face datasets directory")
    parser.add_argument("--export_parquet", action="store_true", default=False,
                        help="Also export .parquet files for backup/hub upload")
    parser.add_argument("--all_cafe_folds", action="store_true", default=False,
                        help="Also export CaFE folds 2-6 as cafe_fold2 ... cafe_fold6")
    args = parser.parse_args()

    data_dir = args.data_dir
    hf_base = args.out_dir
    os.makedirs(hf_base, exist_ok=True)

    print("=" * 60)
    print("EXPORTING TO HUGGING FACE ARROW DATASET")
    print(f"Source Data: {data_dir}")
    print(f"Output Base: {hf_base}")
    print("=" * 60)

    # 1. Combined Multilingual Dataset (Primary for MTKD)
    combined_dir = os.path.join(data_dir, "combined")
    if os.path.isdir(combined_dir):
        print("\n1. Exporting Combined Multilingual Dataset (EN + FR)...")
        combined_splits = {
            "train": os.path.join(combined_dir, "train.csv"),
            "dev":   os.path.join(combined_dir, "dev.csv"),
            "test":  os.path.join(combined_dir, "test.csv"),
        }
        out_comb = os.path.join(hf_base, "combined")
        export_dataset_dict(combined_splits, data_dir, out_comb, export_parquet=args.export_parquet)
        verify_hf_dataset(out_comb)

    # 2. IEMOCAP Sessions (Session 1 to 5)
    print("\n2. Exporting IEMOCAP Datasets (5-fold leave-one-session-out)...")
    for session_idx in range(1, 6):
        sess_dir = os.path.join(data_dir, "en", "iemocap", f"session{session_idx}")
        if os.path.isdir(sess_dir):
            sess_splits = {
                "train": os.path.join(sess_dir, "train.csv"),
                "dev":   os.path.join(sess_dir, "dev.csv"),
                "test":  os.path.join(sess_dir, "test.csv"),
            }
            out_sess = os.path.join(hf_base, f"iemocap_session{session_idx}")
            print(f"\n  Exporting Session {session_idx}...")
            export_dataset_dict(sess_splits, data_dir, out_sess, export_parquet=args.export_parquet)

    # 3. CaFE Dataset (French Monolingual)
    cafe_splits_dir = os.path.join(data_dir, "fr", "cafe", "CaFE_splits")
    if os.path.isdir(cafe_splits_dir):
        print("\n3. Exporting CaFE Dataset (French)...")
        cafe_splits = {
            "train": os.path.join(cafe_splits_dir, "train.csv"),
            "dev":   os.path.join(cafe_splits_dir, "dev.csv"),
            "test":  os.path.join(cafe_splits_dir, "test.csv"),
        }
        out_cafe = os.path.join(hf_base, "cafe")
        export_dataset_dict(cafe_splits, data_dir, out_cafe, export_parquet=args.export_parquet)
        verify_hf_dataset(out_cafe)

        if args.all_cafe_folds:
            for fold in range(2, 7):
                fold_dir = os.path.join(cafe_splits_dir, f"fold{fold}")
                if not os.path.isdir(fold_dir):
                    continue
                print(f"\n  Exporting CaFE fold {fold}...")
                fold_splits = {s: os.path.join(fold_dir, f"{s}.csv") for s in ("train", "dev", "test")}
                export_dataset_dict(fold_splits, data_dir, os.path.join(hf_base, f"cafe_fold{fold}"),
                                    export_parquet=args.export_parquet)

    print("\n" + "=" * 60)
    print("ALL HUGGING FACE ARROW DATASETS EXPORTED SUCCESSFULLY!")
    print(f"Location: {hf_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
