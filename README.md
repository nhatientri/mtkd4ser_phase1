# MTKD4SER: Multi-Teacher Knowledge Distillation for Speech Emotion Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/mtkd4ser_phase1/blob/main/notebooks/MTKD4SER_Google_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers%20%26%20Datasets-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An optimized, language-aware **Multi-Teacher Knowledge Distillation (MTKD)** framework for **Speech Emotion Recognition (SER)**. This codebase is fully engineered for research reproducibility and optimized for execution on **Google Colab Free Tier (Nvidia T4 GPU / 15 GB VRAM)**.

---

## Highlights & Innovations

- **Language-Aware Distillation Routing**: Supports 4 distinct teacher distillation routing mechanisms:
  - `language` *(Oracle)*: Routes student samples directly to the native-language teacher model using sample metadata (`language_id: 0=EN, 2=FR`).
  - `uniform`: Balanced baseline weighting across all active teachers ($1/N$).
  - `cosine`: Dynamic temperature-scaled cosine similarity between student and teacher representations.
  - `confidence`: Softmax probability confidence routing based on teacher classification entropy.
- **Dynamic Dual-Teacher Architecture (EN + FR)**:
  - Dynamically instantiates and computes forward passes only for active teachers (bilingual English + French by default).
  - Eliminates dummy teacher duplication, saving **~33% training time** and **~380 MB–1.27 GB VRAM**, making larger backbones like `wav2vec2-xls-r-300m` runnable on Google Colab T4.
- **Zero Speaker Leakage**:
  - Replaces the historical `dev = test` evaluation flaw with strictly speaker-independent validation splits (LOSO on IEMOCAP and Leave-Two-Speakers-Out on CaFE).
- **Hugging Face Arrow Integration (Zero-Copy)**:
  - Preprocesses audio into Hugging Face Apache Arrow tables (`data/hf_datasets/`).
  - Achieves zero-copy memory mapping (`mmap`), eliminating Google Colab / Google Drive disk I/O bottlenecks during epoch training.

---

## Repository Structure

```
mtkd4ser_phase1/
├── notebooks/
│   └── MTKD4SER_Google_Colab.ipynb # Interactive Colab pipeline notebook
├── main.py                         # CLI entry point for training and evaluation
├── data.py                         # Dataset loaders (IEMOCAP, CaFE, Multilingual)
├── models.py                       # Model factories for FT, Single-KD, and Dynamic MTKD
├── pipeline.py                     # Training loops, validation checkpointing, early stopping
├── utils.py                        # Losses, dynamic routing calculations, metrics, collate fn
├── export_to_hf_dataset.py         # Converts raw WAV/CSV splits to Hugging Face Arrow format
├── reformat_datasets.py            # Audio resampling (16kHz), 4-class normalization, split generator
├── requirements.txt                # Pip package dependencies
├── environment.yml                 # Conda environment definition
├── .gitignore                      # Git ignore rules (protects data and model checkpoints)
└── LICENSE                         # MIT Open Source License
```

---

## Quickstart on Google Colab

You can run the entire pipeline directly in your browser using Google Colab Free:

1. Open [`notebooks/MTKD4SER_Google_Colab.ipynb`](notebooks/MTKD4SER_Google_Colab.ipynb) in Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Follow the step-by-step cells for environment setup, dataset loading, and training.

---

## Local Installation

### Prerequisites
- Python 3.10 or 3.11
- CUDA 11.8+ / 12.x (for GPU training) or Apple Silicon / CPU

### Using Pip
```bash
git clone https://github.com/username/mtkd4ser_phase1.git
%cd mtkd4ser_phase1
pip install -r requirements.txt
```

### Using Conda
```bash
conda env create -f environment.yml
conda activate ser
```

---

## Dataset Preparation

This framework evaluates on 4 common emotion classes:
- `0`: Anger
- `1`: Happiness (*merged with excitement*)
- `2`: Neutral
- `3`: Sadness

### Standard Preprocessing
To convert raw **IEMOCAP** and **CaFE** audio files into standardized 16 kHz mono WAVs with zero speaker leakage splits:

```bash
python reformat_datasets.py \
    --iemocap_in /path/to/IEMOCAP_full_release \
    --cafe_in /path/to/CaFE_48k \
    --out_dir ./data
```

### Export to Hugging Face Arrow (Local Zero-Copy)
```bash
python export_to_hf_dataset.py \
    --data_dir ./data \
    --out_dir ./data/hf_datasets
```

### Push to Hugging Face Hub (Cloud-Native Workflow)
To bypass Google Drive archives entirely and stream data directly to Colab:
```bash
# Upload local Arrow datasets to your private Hugging Face repo
python upload_to_hf.py --repo_id <your-hf-username>/mtkd4ser-phase1
```
Then simply pass `--HF_DATASET_REPO <your-hf-username>/mtkd4ser-phase1` to any training command!

---

## Usage Guide

### 1. Train Monolingual Teacher Models

Fine-tune (FT) baseline teachers on English and French datasets:

```bash
# English Teacher (IEMOCAP Session 1 Fold)
python main.py --PARADIGM FT --LANGUAGE EN --LINGUALITY Monolingual \
    --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 8 --LEARNING_RATE 3e-5 \
    --TRAINING 1 --SAVE_DIR ./checkpoints

# French Teacher (CaFE Fold 1)
python main.py --PARADIGM FT --LANGUAGE FR --LINGUALITY Monolingual \
    --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 8 --LEARNING_RATE 3e-5 \
    --TRAINING 1 --SAVE_DIR ./checkpoints
```

### 2. Train Multi-Teacher Knowledge Distillation (MTKD)

Distill knowledge into a multilingual student model using active English and French teachers:

#### Oracle Language Routing (Recommended)
```bash
python main.py --PARADIGM MTKD --LANGUAGE EN --LINGUALITY Multilingual \
    --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 6 --LEARNING_RATE 3e-5 \
    --TRAINING 1 --ROUTING language \
    --TEACHER_EN_CKPT ./checkpoints/FT_Monolingual_EN_S1.pth \
    --TEACHER_FR_CKPT ./checkpoints/FT_Monolingual_FR_S1.pth \
    --INCLUDE_FI 0 --SAVE_DIR ./checkpoints
```

#### Routing Ablations
You can switch the `--ROUTING` flag to test different distillation mechanisms:
- `--ROUTING uniform`: 50% English + 50% French distillation weight.
- `--ROUTING cosine`: Cosine similarity between student and teacher representations.
- `--ROUTING confidence`: Weight based on teacher softmax prediction certainty.

### 3. Evaluate a Trained Checkpoint

Evaluate any checkpoint on the speaker-independent test set without training:

```bash
python main.py --PARADIGM MTKD --LANGUAGE EN --LINGUALITY Multilingual \
    --SESSION 1 --TRAINING 0 --ROUTING language \
    --TEACHER_EN_CKPT ./checkpoints/FT_Monolingual_EN_S1.pth \
    --TEACHER_FR_CKPT ./checkpoints/FT_Monolingual_FR_S1.pth \
    --INCLUDE_FI 0 --SAVE_DIR ./checkpoints
```

---

## Recommended Batch Sizes (Colab T4 15 GB VRAM)

| Backbone | FT / KD Batch Size | MTKD (2 Teachers: EN + FR) | MTKD (3 Teachers: EN + FI + FR) |
| :--- | :---: | :---: | :---: |
| `facebook/wav2vec2-base` | 8 – 16 | 6 – 12 | 4 – 8 |
| `facebook/wav2vec2-xls-r-300m` | 4 – 8 | 4 – 6 | 2 – 4 |

---

## Experiment Monitoring

Monitor loss curves, Unweighted Recall (UR), and Weighted Recall (WR) via TensorBoard:

```bash
tensorboard --logdir ./checkpoints/tb
```

---

## License

This project is licensed under the [MIT License](LICENSE).
