# MTKD4SER Dataset Directory Structure

This folder holds the preprocessed datasets and Arrow memory-mapped artifacts for Speech Emotion Recognition (SER).
Due to size constraints and licensing, dataset audio files and Arrow shards are excluded from Git via `.gitignore`.

---

## Recommended Directory Layout

```
data/
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
└── hf_datasets/                      # Hugging Face Arrow datasets (Zero-copy loading)
    ├── combined/                     # Multilingual DatasetDict (train, dev, test)
    ├── iemocap_session1/ ... session5/
    └── cafe/
```

---

## How to Prepare the Datasets

### Option 1: Using Raw Datasets (IEMOCAP + CaFE)
1. Download **CaFE** from Zenodo (`CaFE_48k.zip`) and extract it.
2. Download **IEMOCAP** (`IEMOCAP_full_release.tar.gz` or Kaggle release) and extract it.
3. Run the automated reformatting script:
   ```bash
   python reformat_datasets.py \
       --iemocap_in /path/to/IEMOCAP_full_release \
       --cafe_in /path/to/CaFE_48k \
       --out_dir ./data
   ```
4. Convert CSV/WAV splits to Hugging Face Arrow datasets for fast training:
   ```bash
   python export_to_hf_dataset.py --data_dir ./data --out_dir ./data/hf_datasets
   ```

### Option 2: Pre-packaged Colab / Drive Archive
If you packaged `data` as `data_ser_phase1.tar.gz` on Google Drive:
```bash
tar -xzf /content/drive/MyDrive/data_ser_phase1.tar.gz -C ./
```
The codebase will auto-detect `./data/hf_datasets` and load in under 0.2 seconds!

### Option 3: Hugging Face Hub (Cloud Native - No Google Drive needed!)
1. Upload local Arrow datasets to your private Hugging Face repo:
   ```bash
   python upload_to_hf.py --repo_id <your-hf-username>/mtkd4ser-phase1
   ```
2. In Google Colab or on any machine, simply run:
   ```bash
   python main.py ... --HF_DATASET_REPO <your-hf-username>/mtkd4ser-phase1
   ```
   Or pre-cache to local disk in Colab in ~20 seconds:
   ```python
   from datasets import load_dataset
   for cfg in ["combined", "iemocap_session1", "cafe"]:
       ds = load_dataset("<your-hf-username>/mtkd4ser-phase1", cfg)
       ds.save_to_disk(f"data/hf_datasets/{cfg}")
   ```
