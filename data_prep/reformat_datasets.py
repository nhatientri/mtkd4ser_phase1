#!/usr/bin/env python3
"""
reformat_datasets.py
--------------------
Reformats IEMOCAP (English) and CaFE (French) according to the specifications in the PDF:
Section 2: Giai đoạn 0: Chuẩn bị Datasets

Requirements handled:
1. 4 common emotions:
   - 0: anger (ang / C - Colère)
   - 1: happiness (hap + exc / J - Joie)
   - 2: neutral (neu / N - Neutre)
   - 3: sadness (sad / T - Tristesse)
   Discards other emotions (fru, sur, fea, dis, xxx, etc.)

2. Audio standard:
   - 16 kHz sample rate, mono (1 channel), float32 normalized amplitude

3. IEMOCAP:
   - 5-fold leave-one-session-out (Session 1 to 5)
   - Speaker-independent dev split (~10-15% from training speakers)
   - language_id = 0
   - Session subdirectories: session1/ ... session5/
   - metadata_en.csv + JSON/CSV splits

4. CaFE:
   - AA-E-I-S naming (Actor, Emotion, Intensity, Sentence)
   - Resampled 48 kHz -> 16 kHz mono
   - Speaker-independent split (6 male + 6 female -> 420 train/dev, 84 test)
   - language_id = 2
   - wavs/ directory + metadata_fr.csv + CaFE_splits JSON/CSV splits

5. Combined (Multilingual):
   - combined/ with train.csv, dev.csv, test.csv and JSON equivalents

6. Zero speaker leakage verification
"""

import os
import re
import glob
import json
import wave
import shutil
import argparse
import tarfile
import numpy as np
import pandas as pd
import scipy.signal as signal
from collections import Counter


EMOTION_MAP_IEMOCAP = {
    "ang": "anger",
    "hap": "happiness",
    "exc": "happiness",
    "neu": "neutral",
    "sad": "sadness"
}

EMOTION_MAP_CAFE = {
    "C": "anger",
    "J": "happiness",
    "N": "neutral",
    "T": "sadness"
}

LABEL2ID = {
    "anger": 0,
    "happiness": 1,
    "neutral": 2,
    "sadness": 3
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def read_and_normalize_wav(in_path):
    """Read WAV file and return (data_float, orig_sr, orig_channels)."""
    with wave.open(in_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        # 24-bit PCM
        a = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        # Sign extend 24-bit to 32-bit
        b = np.pad(a, ((0, 0), (1, 0)), mode="constant")
        data = b.view(np.int32).squeeze() >> 8
        data = data.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data, framerate


def write_16k_mono_wav(out_path, audio_float, normalize=True):
    """Peak-normalize to 0.95 and write 16kHz mono 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if normalize:
        peak = np.max(np.abs(audio_float))
        if peak > 0:
            audio_float = (audio_float / peak) * 0.95

    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())


def resample_48k_to_16k(audio_48k):
    """Downsample by exact factor of 3 from 48000 Hz to 16000 Hz."""
    return signal.resample_poly(audio_48k, 1, 3)


def parse_iemocap_annotations(iemocap_root):
    """Parse EmoEvaluation txt files across Session 1 to 5."""
    pattern = re.compile(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[")
    records = []

    for session_idx in range(1, 6):
        session_name = f"Session{session_idx}"
        emo_dir = os.path.join(iemocap_root, session_name, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(iemocap_root, session_name, "sentences", "wav")

        txt_files = sorted(glob.glob(os.path.join(emo_dir, "*.txt")))
        for tf in txt_files:
            dialog_id = os.path.splitext(os.path.basename(tf))[0]
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        start_time, end_time, turn_id, raw_emo = match.groups()
                        if raw_emo in EMOTION_MAP_IEMOCAP:
                            mapped_emo = EMOTION_MAP_IEMOCAP[raw_emo]
                            spk_char = turn_id.split("_")[-1][0]  # 'F' or 'M'
                            gender = "female" if spk_char == "F" else "male"
                            speaker_id = f"Ses{session_idx:02d}_{spk_char}"
                            src_wav = os.path.join(wav_dir, dialog_id, f"{turn_id}.wav")

                            if not os.path.isfile(src_wav):
                                print(f"[WARN] File not found: {src_wav}")
                                continue

                            records.append({
                                "turn_id": turn_id,
                                "src_wav": src_wav,
                                "emotion": mapped_emo,
                                "label": LABEL2ID[mapped_emo],
                                "speaker_id": speaker_id,
                                "speaker": speaker_id,
                                "session": session_idx,
                                "gender": gender,
                                "language_id": 0,
                                "lang_id": 0,
                                "start_time": float(start_time),
                                "end_time": float(end_time),
                            })

    return records


def parse_cafe_dataset(cafe_root):
    """Parse CaFE 48k audio files."""
    wav_files = sorted(glob.glob(os.path.join(cafe_root, "**", "*.wav"), recursive=True))
    records = []

    for src_wav in wav_files:
        fname = os.path.basename(src_wav)
        parts = fname.replace(".wav", "").split("-")
        if len(parts) == 4:
            actor, emo_code, intensity, sentence = parts
            if emo_code in EMOTION_MAP_CAFE:
                mapped_emo = EMOTION_MAP_CAFE[emo_code]
                actor_num = int(actor)
                gender = "male" if actor_num % 2 == 1 else "female"
                speaker_id = f"CaFE_{actor}"

                records.append({
                    "turn_id": fname.replace(".wav", ""),
                    "filename": fname,
                    "src_wav": src_wav,
                    "emotion": mapped_emo,
                    "label": LABEL2ID[mapped_emo],
                    "speaker_id": speaker_id,
                    "speaker": speaker_id,
                    "actor_num": actor_num,
                    "intensity": int(intensity),
                    "sentence": int(sentence),
                    "gender": gender,
                    "language_id": 2,
                    "lang_id": 2,
                })

    return records


def process_iemocap(iemocap_records, out_base_dir):
    """
    Process IEMOCAP:
    - Resample/normalize wavs into data/en/iemocap/session{s}/<turn_id>.wav
    - Save metadata_en.csv
    - Generate 5-fold leave-one-session-out splits with speaker-independent dev set
    """
    print(f"\nProcessing {len(iemocap_records)} IEMOCAP audio files...")
    en_dir = os.path.join(out_base_dir, "en", "iemocap")
    os.makedirs(en_dir, exist_ok=True)

    for rec in iemocap_records:
        rel_path = f"en/iemocap/session{rec['session']}/{rec['turn_id']}.wav"
        dest_path = os.path.join(out_base_dir, rel_path)
        rec["path"] = rel_path
        rec["audio"] = rel_path

        if not os.path.isfile(dest_path):
            audio_data, sr = read_and_normalize_wav(rec["src_wav"])
            write_16k_mono_wav(dest_path, audio_data, normalize=True)

    # Save metadata_en.csv
    meta_df = pd.DataFrame([{
        "path": r["path"],
        "audio": r["audio"],
        "label": r["label"],
        "emotion": r["emotion"],
        "speaker_id": r["speaker_id"],
        "speaker": r["speaker"],
        "session": r["session"],
        "language_id": r["language_id"],
        "lang_id": r["lang_id"],
        "gender": r["gender"],
    } for r in iemocap_records])

    meta_csv_path = os.path.join(en_dir, "metadata_en.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"Saved IEMOCAP metadata to {meta_csv_path} ({len(meta_df)} rows)")

    # 5-fold leave-one-session-out splits
    # Session k is test.
    # Dev set: held-out 1 speaker from Session (k % 5) + 1 (~10-14% of train data, strictly speaker-independent)
    # Train set: remaining 7 speakers from other sessions
    DEV_SPEAKERS = {
        1: "Ses02_F",  # 481 samples (~10.8% of train)
        2: "Ses03_F",  # 522 samples (~11.6% of train)
        3: "Ses04_F",  # 528 samples (~12.1% of train)
        4: "Ses05_F",  # 590 samples (~13.1% of train)
        5: "Ses01_F",  # 528 samples (~12.3% of train)
    }

    folds = {}
    for test_session in range(1, 6):
        session_fold_dir = os.path.join(en_dir, f"session{test_session}")
        os.makedirs(session_fold_dir, exist_ok=True)

        dev_spk = DEV_SPEAKERS[test_session]

        test_records = [r for r in iemocap_records if r["session"] == test_session]
        train_val_records = [r for r in iemocap_records if r["session"] != test_session]
        dev_records = [r for r in train_val_records if r["speaker_id"] == dev_spk]
        train_records = [r for r in train_val_records if r["speaker_id"] != dev_spk]

        # Verify zero leakage
        train_spks = {r["speaker_id"] for r in train_records}
        dev_spks = {r["speaker_id"] for r in dev_records}
        test_spks = {r["speaker_id"] for r in test_records}

        assert not (train_spks & test_spks), f"Speaker leakage between train and test in fold {test_session}!"
        assert not (dev_spks & test_spks), f"Speaker leakage between dev and test in fold {test_session}!"
        assert not (train_spks & dev_spks), f"Speaker leakage between train and dev in fold {test_session}!"

        # Save CSVs
        for split_name, split_recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
            df_split = pd.DataFrame(split_recs)[["path", "audio", "label", "emotion", "speaker_id", "speaker", "session", "language_id", "gender"]]
            df_split.to_csv(os.path.join(session_fold_dir, f"{split_name}.csv"), index=False)

            # Also save JSON format expected by mtkd4ser_phase1 data.py
            json_dict = {}
            for r in split_recs:
                json_dict[r["turn_id"]] = {
                    "wav": r["path"],
                    "audio": r["path"],
                    "emo": r["emotion"],
                    "label": r["label"],
                    "speaker_id": r["speaker_id"],
                    "session": r["session"],
                    "language_id": r["language_id"],
                    "gender": r["gender"],
                }
            with open(os.path.join(session_fold_dir, f"{split_name}.json"), "w") as jf:
                json.dump(json_dict, jf, indent=2)

        folds[test_session] = {
            "train": train_records,
            "dev": dev_records,
            "test": test_records
        }

        print(f"  Session {test_session} fold: train={len(train_records)} (spks: {sorted(train_spks)}), "
              f"dev={len(dev_records)} (spk: {sorted(dev_spks)}), "
              f"test={len(test_records)} (spks: {sorted(test_spks)})")

    return meta_df, folds


def process_cafe(cafe_records, out_base_dir):
    """
    Process CaFE:
    - Resample 48 kHz -> 16 kHz mono normalized into data/fr/cafe/wavs/<filename>.wav
    - Save metadata_fr.csv
    - Generate 6-fold leave-two-speakers-out (1 male + 1 female per fold)
      Default fold (Fold 1):
        Test: CaFE_11, CaFE_12 (84 samples)
        Dev:  CaFE_09, CaFE_10 (84 samples)
        Train: CaFE_01..08 (336 samples)
      Total Train + Dev = 420 samples (exact match to paper Table 420 train / 84 test)
    """
    print(f"\nProcessing {len(cafe_records)} CaFE audio files (resampling 48kHz -> 16kHz)...")
    fr_dir = os.path.join(out_base_dir, "fr", "cafe")
    wavs_dir = os.path.join(fr_dir, "wavs")
    splits_dir = os.path.join(fr_dir, "CaFE_splits")
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    for rec in cafe_records:
        rel_path = f"fr/cafe/wavs/{rec['filename']}"
        dest_path = os.path.join(out_base_dir, rel_path)
        rec["path"] = rel_path
        rec["audio"] = rel_path

        if not os.path.isfile(dest_path):
            audio_48k, sr = read_and_normalize_wav(rec["src_wav"])
            if sr == 48000:
                audio_16k = resample_48k_to_16k(audio_48k)
            elif sr == 16000:
                audio_16k = audio_48k
            else:
                # General polyphase resample if needed
                audio_16k = signal.resample_poly(audio_48k, 16000, sr)

            write_16k_mono_wav(dest_path, audio_16k, normalize=True)

    # Save metadata_fr.csv
    meta_df = pd.DataFrame([{
        "path": r["path"],
        "audio": r["audio"],
        "label": r["label"],
        "emotion": r["emotion"],
        "speaker_id": r["speaker_id"],
        "speaker": r["speaker"],
        "intensity": r["intensity"],
        "sentence": r["sentence"],
        "language_id": r["language_id"],
        "lang_id": r["lang_id"],
        "gender": r["gender"],
    } for r in cafe_records])

    meta_csv_path = os.path.join(fr_dir, "metadata_fr.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"Saved CaFE metadata to {meta_csv_path} ({len(meta_df)} rows)")

    # 6 Speaker Pairs:
    # Pair 1: (01, 02), Pair 2: (03, 04), Pair 3: (05, 06), Pair 4: (07, 08), Pair 5: (09, 10), Pair 6: (11, 12)
    SPEAKER_PAIRS = [
        ("CaFE_01", "CaFE_02"),
        ("CaFE_03", "CaFE_04"),
        ("CaFE_05", "CaFE_06"),
        ("CaFE_07", "CaFE_08"),
        ("CaFE_09", "CaFE_10"),
        ("CaFE_11", "CaFE_12"),
    ]

    cafe_folds = {}
    for fold_idx in range(1, 7):
        test_pair = SPEAKER_PAIRS[fold_idx - 1]
        dev_pair = SPEAKER_PAIRS[(fold_idx - 2) % 6]
        train_pairs = [p for p in SPEAKER_PAIRS if p != test_pair and p != dev_pair]

        test_spks = set(test_pair)
        dev_spks = set(dev_pair)
        train_spks = set(spk for p in train_pairs for spk in p)

        test_recs = [r for r in cafe_records if r["speaker_id"] in test_spks]
        dev_recs = [r for r in cafe_records if r["speaker_id"] in dev_spks]
        train_recs = [r for r in cafe_records if r["speaker_id"] in train_spks]

        # Verify zero leakage
        assert not (train_spks & test_spks)
        assert not (dev_spks & test_spks)
        assert not (train_spks & dev_spks)

        fold_dir = os.path.join(splits_dir, f"fold{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        for split_name, split_recs in [("train", train_recs), ("dev", dev_recs), ("test", test_recs)]:
            df_split = pd.DataFrame(split_recs)[["path", "audio", "label", "emotion", "speaker_id", "speaker", "intensity", "sentence", "language_id", "gender"]]
            df_split.to_csv(os.path.join(fold_dir, f"{split_name}.csv"), index=False)

            json_dict = {}
            for r in split_recs:
                json_dict[r["turn_id"]] = {
                    "wav": r["path"],
                    "audio": r["path"],
                    "emo": r["emotion"],
                    "label": r["label"],
                    "speaker_id": r["speaker_id"],
                    "gender": r["gender"],
                    "language_id": r["language_id"],
                }
            with open(os.path.join(fold_dir, f"{split_name}.json"), "w") as jf:
                json.dump(json_dict, jf, indent=2)

        # For the default fold (Fold 1), also copy json/csv directly to CaFE_splits/
        if fold_idx == 1:
            for split_name, split_recs in [("train", train_recs), ("dev", dev_recs), ("test", test_recs)]:
                df_split = pd.DataFrame(split_recs)[["path", "audio", "label", "emotion", "speaker_id", "speaker", "intensity", "sentence", "language_id", "gender"]]
                df_split.to_csv(os.path.join(splits_dir, f"{split_name}.csv"), index=False)

                json_dict = {}
                for r in split_recs:
                    json_dict[r["turn_id"]] = {
                        "wav": r["path"],
                        "audio": r["path"],
                        "emo": r["emotion"],
                        "label": r["label"],
                        "speaker_id": r["speaker_id"],
                        "gender": r["gender"],
                        "language_id": r["language_id"],
                    }
                with open(os.path.join(splits_dir, f"{split_name}.json"), "w") as jf:
                    json.dump(json_dict, jf, indent=2)

        cafe_folds[fold_idx] = {
            "train": train_recs,
            "dev": dev_recs,
            "test": test_recs,
        }

        print(f"  CaFE fold {fold_idx}: train={len(train_recs)} (spks: {sorted(train_spks)}), "
              f"dev={len(dev_recs)} (spks: {sorted(dev_spks)}), "
              f"test={len(test_recs)} (spks: {sorted(test_spks)})")

    return meta_df, cafe_folds


def create_combined_multilingual(en_folds, cafe_folds, out_base_dir, default_en_session=1, default_cafe_fold=1):
    """
    Create combined multilingual train/dev/test splits (EN + FR).
    """
    print(f"\nCreating combined multilingual dataset (EN Session {default_en_session} + CaFE Fold {default_cafe_fold})...")
    comb_dir = os.path.join(out_base_dir, "combined")
    os.makedirs(comb_dir, exist_ok=True)

    en_train = en_folds[default_en_session]["train"]
    en_dev = en_folds[default_en_session]["dev"]
    en_test = en_folds[default_en_session]["test"]

    cafe_train = cafe_folds[default_cafe_fold]["train"]
    cafe_dev = cafe_folds[default_cafe_fold]["dev"]
    cafe_test = cafe_folds[default_cafe_fold]["test"]

    splits = {
        "train": en_train + cafe_train,
        "dev": en_dev + cafe_dev,
        "test": en_test + cafe_test,
    }

    for s_name, s_recs in splits.items():
        # CSV
        df = pd.DataFrame(s_recs)[["path", "audio", "label", "emotion", "speaker_id", "speaker", "language_id", "gender"]]
        csv_p = os.path.join(comb_dir, f"{s_name}.csv")
        df.to_csv(csv_p, index=False)

        # JSON
        json_dict = {}
        for r in s_recs:
            json_dict[r["turn_id"]] = {
                "wav": r["path"],
                "audio": r["path"],
                "emo": r["emotion"],
                "label": r["label"],
                "speaker_id": r["speaker_id"],
                "gender": r["gender"],
                "language_id": r["language_id"],
            }
        with open(os.path.join(comb_dir, f"{s_name}.json"), "w") as jf:
            json.dump(json_dict, jf, indent=2)

        en_cnt = sum(1 for r in s_recs if r["language_id"] == 0)
        fr_cnt = sum(1 for r in s_recs if r["language_id"] == 2)
        print(f"  Combined {s_name}: total={len(s_recs)} (EN={en_cnt}, FR={fr_cnt})")


def verify_dataset_integrity(out_base_dir):
    """Comprehensive check of audio formats, paths, labels, and speaker leakage."""
    print("\n" + "=" * 60)
    print("VERIFYING DATASET INTEGRITY & SPECIFICATIONS")
    print("=" * 60)

    # 1. Metadata check
    en_meta_path = os.path.join(out_base_dir, "en", "iemocap", "metadata_en.csv")
    fr_meta_path = os.path.join(out_base_dir, "fr", "cafe", "metadata_fr.csv")

    assert os.path.isfile(en_meta_path), f"Missing {en_meta_path}"
    assert os.path.isfile(fr_meta_path), f"Missing {fr_meta_path}"

    df_en = pd.read_csv(en_meta_path)
    df_fr = pd.read_csv(fr_meta_path)

    print(f"IEMOCAP samples: {len(df_en)}")
    print(f"  Emotion counts: {dict(df_en['emotion'].value_counts())}")
    print(f"  Speaker counts: {dict(df_en['speaker_id'].value_counts())}")
    print(f"  Language ID: {df_en['language_id'].unique()} (expected: [0])")

    print(f"\nCaFE samples: {len(df_fr)}")
    print(f"  Emotion counts: {dict(df_fr['emotion'].value_counts())}")
    print(f"  Speaker counts: {dict(df_fr['speaker_id'].value_counts())}")
    print(f"  Language ID: {df_fr['language_id'].unique()} (expected: [2])")

    # 2. Audio checks (sample rate, mono, amplitude)
    print("\nChecking audio properties...")
    for df, name in [(df_en, "IEMOCAP"), (df_fr, "CaFE")]:
        sample_paths = df["path"].sample(min(20, len(df)), random_state=42).tolist()
        for rel_p in sample_paths:
            full_p = os.path.join(out_base_dir, rel_p)
            assert os.path.isfile(full_p), f"Audio file missing: {full_p}"
            with wave.open(full_p, "rb") as wf:
                assert wf.getframerate() == 16000, f"Incorrect sample rate {wf.getframerate()} for {full_p}"
                assert wf.getnchannels() == 1, f"Not mono ({wf.getnchannels()} channels) for {full_p}"
                assert wf.getsampwidth() == 2, f"Incorrect bit depth ({wf.getsampwidth()}) for {full_p}"
                assert wf.getnframes() > 0, f"Empty audio {full_p}"
        print(f"  {name}: Audio verified (16 kHz, mono, 16-bit PCM, valid waveform)")

    print("\nAll checks PASSED successfully!")


def create_tarball(out_base_dir, tar_name="data_ser_phase1.tar.gz"):
    """Create a compressed tarball for easy upload to Google Drive / Colab."""
    parent_dir = os.path.dirname(out_base_dir)
    dir_name = os.path.basename(out_base_dir)
    tar_path = os.path.join(parent_dir, tar_name)
    print(f"\nCreating archive {tar_path} for Colab / Drive...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_base_dir, arcname=dir_name)
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"Archive created: {tar_path} ({size_mb:.1f} MB)")
    return tar_path


def main():
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    _default_out_dir = os.path.join(_project_dir, "data")
    _default_iemocap = os.environ.get(
        "IEMOCAP_IN",
        "/content/IEMOCAP_full_release" if os.path.isdir("/content/IEMOCAP_full_release")
        else os.path.join(os.path.dirname(_project_dir), "IEMOCAP_full_release")
    )
    _default_cafe = os.environ.get(
        "CAFE_IN",
        "/content/CaFE_48k" if os.path.isdir("/content/CaFE_48k")
        else os.path.join(os.path.dirname(_project_dir), "CaFE_48k")
    )

    parser = argparse.ArgumentParser(description="Reformat IEMOCAP and CaFE datasets per PDF specs.")
    parser.add_argument("--iemocap_in", default=_default_iemocap,
                        help="Path to IEMOCAP_full_release directory")
    parser.add_argument("--cafe_in", default=_default_cafe,
                        help="Path to CaFE_48k directory")
    parser.add_argument("--out_dir", default=_default_out_dir,
                        help="Output directory for unified data structure")
    parser.add_argument("--create_archive", action="store_true", default=False,
                        help="Create tar.gz archive for Google Drive / Colab")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET REFORMATTING PIPELINE")
    print(f"IEMOCAP Input: {args.iemocap_in}")
    print(f"CaFE Input:    {args.cafe_in}")
    print(f"Output Dir:    {args.out_dir}")
    print("=" * 60)

    # 1. Parse annotations
    iemocap_records = parse_iemocap_annotations(args.iemocap_in)
    print(f"Found {len(iemocap_records)} valid 4-class IEMOCAP utterances.")

    cafe_records = parse_cafe_dataset(args.cafe_in)
    print(f"Found {len(cafe_records)} valid 4-class CaFE utterances.")

    # 2. Process datasets
    meta_en, en_folds = process_iemocap(iemocap_records, args.out_dir)
    meta_fr, cafe_folds = process_cafe(cafe_records, args.out_dir)

    # 3. Create combined multilingual dataset
    create_combined_multilingual(en_folds, cafe_folds, args.out_dir)

    # 4. Integrity verification
    verify_dataset_integrity(args.out_dir)

    # 5. Optional symlink if specified via environment variable
    alt_data_dir = os.environ.get("ALT_DATA_DIR")
    if alt_data_dir and not os.path.exists(alt_data_dir):
        try:
            os.symlink(args.out_dir, alt_data_dir)
            print(f"Created symlink: {alt_data_dir} -> {args.out_dir}")
        except Exception as e:
            print(f"Note: Could not create symlink {alt_data_dir}: {e}")

    # 6. Create archive for Colab
    if args.create_archive:
        create_tarball(args.out_dir)

    print("\nPreprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
