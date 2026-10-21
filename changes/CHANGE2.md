# Dataset Reformatting Summary (MTKD4SER Phase 1)

This document summarizes the preprocessing and reformatting performed on the **IEMOCAP** (English) and **CaFE** (Canadian French) datasets according to the specifications in **Giai đoạn 0 (Mục 2)** of the research improvement plan.

---

## 1. Unified Standards Applied

| Criteria | IEMOCAP (English) | CaFE (French) | Combined (Multilingual) |
| :--- | :--- | :--- | :--- |
| **Language ID** | `0` | `2` | `0` (EN) and `2` (FR) |
| **Audio Format** | 16 kHz, Mono, 16-bit PCM | 48 kHz $\rightarrow$ **16 kHz, Mono**, 16-bit PCM | 16 kHz, Mono, 16-bit PCM |
| **Amplitude** | Peak normalized to 0.95 | Peak normalized to 0.95 | Peak normalized to 0.95 |
| **Emotion Classes** | 4 common classes | 4 common classes | 4 common classes |
| **Label Mapping** | `{0: anger, 1: happiness, 2: neutral, 3: sadness}` | `{0: anger, 1: happiness, 2: neutral, 3: sadness}` | `{0: anger, 1: happiness, 2: neutral, 3: sadness}` |
| **Classes Merged/Dropped** | `exc` merged into `happiness`<br>Dropped: `fru, sur, fea, oth, xxx, dis` | Dropped: `Peur (fea), Surprise (sur), Dégoût (dis)` | Same |

---

## 2. Dataset Splitting & Speaker Leakage Prevention

### A. IEMOCAP (English) – 5,531 samples across 10 speakers
* **Protocol**: 5-fold Leave-One-Session-Out (LOSO).
* **Fixing the `dev = test` flaw**: Instead of evaluating on the test set during validation, each fold creates a **strictly speaker-independent dev set** using 1 speaker (~10–13% of training data) from another session.
* **Per-fold breakdown**:
  * **Session 1 Fold**: Train = 3,965 (7 spks) | Dev = 481 (`Ses02_F`) | Test = 1,085 (Session 1: `Ses01_F`, `Ses01_M`)
  * **Session 2 Fold**: Train = 3,986 (7 spks) | Dev = 522 (`Ses03_F`) | Test = 1,023 (Session 2: `Ses02_F`, `Ses02_M`)
  * **Session 3 Fold**: Train = 3,852 (7 spks) | Dev = 528 (`Ses04_F`) | Test = 1,151 (Session 3: `Ses03_F`, `Ses03_M`)
  * **Session 4 Fold**: Train = 3,910 (7 spks) | Dev = 590 (`Ses05_F`) | Test = 1,031 (Session 4: `Ses04_F`, `Ses04_M`)
  * **Session 5 Fold**: Train = 3,762 (7 spks) | Dev = 528 (`Ses01_F`) | Test = 1,241 (Session 5: `Ses05_F`, `Ses05_M`)
* **Speaker Leakage**: **0%** between train, dev, and test.

### B. CaFE (Canadian French) – 504 samples across 12 speakers
* **Protocol**: 6-fold Leave-Two-Speakers-Out (1 male + 1 female per fold).
* **Default Fold 1 (Matching paper's 420 train / 84 test)**:
  * **Train**: 336 samples (8 speakers: `CaFE_03` to `CaFE_10`)
  * **Dev**: 84 samples (2 speakers: `CaFE_11`, `CaFE_12`)
  * **Test**: 84 samples (2 speakers: `CaFE_01`, `CaFE_02`)
  * Train + Dev = 420 samples (10 speakers: 5 male + 5 female).
* **Speaker Leakage**: **0%** between train, dev, and test.

### C. Combined (Multilingual) – 6,035 samples
* Combines IEMOCAP Session 1 and CaFE Fold 1 for multilingual student training:
  * **Train**: 4,301 samples (3,965 EN + 336 FR)
  * **Dev**: 565 samples (481 EN + 84 FR)
  * **Test**: 1,169 samples (1,085 EN + 84 FR)

---

## 3. Directory Layout

The generated data is stored under `mtkd4ser_phase1/data/`:

```
mtkd4ser_phase1/data/
├── en/
│   └── iemocap/
│       ├── session1/ ... session5/   # 16kHz WAVs + train/dev/test.csv & .json
│       └── metadata_en.csv           # 5,531 records
├── fr/
│   └── cafe/
│       ├── wavs/                     # 504 resampled 16kHz WAVs
│       ├── CaFE_splits/              # train/dev/test.csv & .json (folds 1-6)
│       └── metadata_fr.csv           # 504 records
├── combined/
│   ├── train.csv / train.json        # 4,301 samples (EN + FR)
│   ├── dev.csv / dev.json            # 565 samples (EN + FR)
│   └── test.csv / test.json          # 1,169 samples (EN + FR)
└── hf_datasets/                      # Hugging Face Arrow datasets (save_to_disk)
    ├── combined/                     # Multilingual DatasetDict
    ├── iemocap_session1/ ... session5/
    └── cafe/
```

* **Symlink**: `/Users/hunter/Downloads/data` $\rightarrow$ `mtkd4ser_phase1/data`.
* **Colab Archive**: `mtkd4ser_phase1/data_ser_phase1.tar.gz` (662.3 MB) ready for upload to Google Drive.

---

## 4. Why Hugging Face Arrow (`hf_datasets/`) was Generated

1. **Zero-Copy Memory Mapping (`mmap`)**: Calling `datasets.load_from_disk(...)` loads datasets in under 0.2 seconds without loading all audio files into RAM.
2. **Eliminating Google Drive I/O Bottleneck**: On Google Colab, reading thousands of individual audio files over Google Drive mounts causes severe I/O stalls. Sharded Arrow files resolve this completely.
3. **Decoded Audio**: Features are registered with `datasets.Audio(sampling_rate=16_000)`, delivering decoded float32 waveforms on-the-fly.

---

## 5. Scripts Created and Updated

1. **`reformat_datasets.py`**:
   * Automated extraction, filtering, 48kHz $\rightarrow$ 16kHz resampling, amplitude normalization, speaker parsing, and CSV/JSON generation.
2. **`export_to_hf_dataset.py`**:
   * Converts the CSV and WAV splits into Hugging Face Arrow format (`save_to_disk`) with full schema definitions.
3. **`data.py` (Updated)**:
   * Auto-detects local `data/` roots.
   * Auto-loads from `data/hf_datasets/` when available for ultra-fast startup.
   * Seamless fallback to JSON/CSV splits.
4. **Environment (`ser`)**:
   * Installed `torch`, `torchaudio`, `transformers`, `datasets`, `librosa`, `soundfile`, `pandas`, `scikit-learn`, `tensorboard`.
