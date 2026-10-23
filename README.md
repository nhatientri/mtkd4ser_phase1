# MTKD4SER: Multi-Teacher Knowledge Distillation for Speech Emotion Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/mtkd4ser_phase1/blob/main/notebooks/MTKD4SER_Google_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers%20%26%20Datasets-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An optimized, language-aware **Multi-Teacher Knowledge Distillation (MTKD)** framework for **Speech Emotion Recognition (SER)**. This codebase is fully engineered for research reproducibility and optimized for execution on **Google Colab Free Tier (Nvidia T4 GPU / 15 GB VRAM)**.

---

## Highlights

- **Per-sample language-aware distillation routing** (`--ROUTING`):
  - `language` *(oracle)*: each clip is distilled only from the teacher of its own language (`language_id`: 0 = EN, 2 = FR).
  - `uniform`: equal weight on every teacher (1/K).
  - `cosine`: per-sample cosine similarity between student and teacher logits, softmax over teachers / `COSINE_TAU`.
  - `confidence`: per-sample teacher confidence (1 − normalised entropy of its prediction), softmax / `CONF_TAU`.
- **Cached teachers**: teachers are frozen, so their logits are computed once, cached in `SAVE_DIR/teacher_cache/`, and the teachers leave GPU memory before the student trains. An MTKD epoch costs about the same as fine-tuning the student alone, whatever the routing mode.
- **Any student size**: distillation matches only the 4 output logits, so the student can be smaller than the teachers (`--BACKBONE`, `--NUM_LAYERS`).
- **Correct padding**: clips are padded per batch with an attention mask, so padding is ignored by attention and by the classifier's mean-pooling.
- **Per-language evaluation**: dev/test report UR for EN and FR separately; multilingual runs pick their best epoch on the mean per-language dev UR.
- **Speaker-independent splits**: leave-one-session-out on IEMOCAP, leave-two-speakers-out on CaFE (6 folds).
- **Efficiency logging**: every run writes `SAVE_DIR/results/<run>.json` (test UR overall/EN/FR, parameters, sec/epoch, samples/s, peak GPU memory, test ms/clip); `summarize_results.py` tabulates them.

---

## Repository Structure

```
mtkd4ser_phase1/
├── main.py                 # CLI entry point: data → (cached teacher logits) → train → test
├── mtkd4ser/               # library code (imported as `from mtkd4ser.<module> import ...`)
│   ├── config.py           # Config dataclass; every field is a --FLAG; run naming
│   ├── constants.py        # labels and language ids (single source of truth)
│   ├── data.py             # IEMOCAP / CaFE / combined loaders, CSV → HF split, waveform preparation
│   ├── models.py           # SERModel wrapper (normalisation + attention mask), checkpoint loading
│   ├── distill.py          # per-sample KD loss and the 4 routers
│   ├── pipeline.py         # one Trainer for FT / KD / MTKD, teacher-logit cache
│   ├── checkpoints.py      # checkpoint format/loading, local path or W&B artifact resolution
│   ├── tracking.py         # W&B run logging (init, teacher lineage, model artifacts, test metrics)
│   └── utils.py            # collate (padding + mask), metrics (incl. per-language UR)
├── summarize_results.py    # compare finished runs (accuracy + efficiency)
├── data_prep/              # one-time dataset preparation (run in this order)
│   ├── reformat_datasets.py    # raw audio → 16 kHz mono, 4 classes, speaker-independent splits
│   ├── export_to_hf_dataset.py # CSV splits → Hugging Face Arrow datasets
│   └── upload_to_hf.py         # push Arrow datasets to a private HF Hub repo
├── changes/                # CHANGE1.md … CHANGE6.md: change log (design decisions, fixed bugs)
├── tests/                  # pytest: routing/loss, batching/metrics, split leakage, offline smoke run
└── notebooks/MTKD4SER_Google_Colab.ipynb
```

Losses (λ = `LAMBDA_KD`, T = `TEMPERATURE`):

```
FT    loss = CE
KD    loss = (1-λ)·CE + λ·T²·KL(teacher ‖ student)
MTKD  loss = (1-λ)·CE + λ·mean_b Σ_k w[b,k]·T²·KL(teacher_k[b] ‖ student[b])     w from the router
```

---

## Installation

```bash
git clone https://github.com/username/mtkd4ser_phase1.git
cd mtkd4ser_phase1
pip install -r requirements.txt        # or: conda env create -f environment.yml && conda activate ser
python -m pytest -q tests              # unit tests + an offline end-to-end smoke run (~1-2 min on CPU)
```

---

## Dataset Preparation

4 classes: `0` anger, `1` happiness (+ excitement), `2` neutral, `3` sadness.

```bash
python data_prep/reformat_datasets.py --iemocap_in /path/to/IEMOCAP_full_release --cafe_in /path/to/CaFE_48k --out_dir ./data
python data_prep/export_to_hf_dataset.py --data_dir ./data --out_dir ./data/hf_datasets [--all_cafe_folds]
python data_prep/upload_to_hf.py --repo_id <user>/mtkd4ser-phase1 [--all_sessions] [--all_cafe_folds]   # for Colab
```

Loaders look for, in order: local Arrow (`data/hf_datasets/`), the reformatted CSV splits, then the HF Hub repo
(`--HF_DATASET_REPO`). `MTKD_DATA_DIR` overrides the `data/` location.

---

## Usage

Run names encode the setup, e.g. `FT_Monolingual_EN_S1_wav2vec2-base_seed42` or
`MTKD_language_Multilingual_ENFR_S1F1_distilhubert_seed42`. Checkpoints: `SAVE_DIR/<run>.pth` (best) and `<run>_last.pth`.
A run starts fresh unless you pass `--RESUME 1` or `--RESUME_CKPT <path | W&B artifact>`.

### 1. Teachers and baselines
```bash
python main.py --PARADIGM FT --LANGUAGE EN --SESSION 1          # English teacher
python main.py --PARADIGM FT --LANGUAGE FR --CAFE_FOLD 1        # French teacher
python main.py --PARADIGM FT --LINGUALITY Multilingual          # multilingual FT baseline (no teachers)
```

### 2. MTKD
```bash
python main.py --PARADIGM MTKD --LINGUALITY Multilingual --ROUTING language \
    --TEACHER_EN_CKPT checkpoints/FT_Monolingual_EN_S1_wav2vec2-base_seed42.pth \
    --TEACHER_FR_CKPT checkpoints/FT_Monolingual_FR_F1_wav2vec2-base_seed42.pth
# ablations: --ROUTING uniform | cosine | confidence     language-balanced sampling: --BALANCE_LANG 1
```

### 3. Smaller students (fast iteration)

| Student | Params | Approx. training compute per clip vs. full student* |
| :--- | ---: | ---: |
| `--BACKBONE facebook/wav2vec2-base` | ~95M | 1× |
| `--BACKBONE facebook/wav2vec2-base --NUM_LAYERS 6` | ~52M | ~0.6× |
| `--BACKBONE facebook/wav2vec2-base --NUM_LAYERS 4` | ~38M | ~0.45× |
| `--BACKBONE ntu-spml/distilhubert` | ~24M | ~0.35× |

\*Estimated from FLOPs (frozen CNN front-end forward + transformer forward/backward); measure the real numbers with
`summarize_results.py`. Smaller students usually need a slightly higher learning rate (e.g. 5e-5 – 1e-4).
Choose routing methods on a small student, then confirm the winner with the full-size one.

### 4. Evaluate and compare
```bash
python main.py --PARADIGM MTKD --LINGUALITY Multilingual --TRAINING 0 --STUDENT_CKPT <path | W&B artifact>
python summarize_results.py --SAVE_DIR checkpoints            # one row per run
python summarize_results.py --SAVE_DIR checkpoints --group    # mean ± std over seeds and folds
```

For reportable numbers, run all IEMOCAP sessions (`--SESSION 1..5`) and CaFE folds (`--CAFE_FOLD 1..6`) with 2–3 seeds.

### Useful flags
`--AMP 1` fp16 on CUDA · `--ACCUM_STEPS` gradient accumulation · `--WARMUP_RATIO` · `--MAX_DURATION` clip length cap ·
`--SELECT_METRIC auto|UR|UR_lang_mean|WR` · `--SAVE_ALL_EPOCHS 1` keep every epoch (≈1.1 GB each for wav2vec2-base) ·
`--USE_WANDB 1` Weights & Biases logging · `--TIME_BUDGET_MIN 20` stop after 20 min of training, with warmup/decay
following the clock instead of `N_EPOCHS` (fixed-compute comparisons, see `changes/CHANGE7.md`).

### Running on Colab from VS Code
- **Colab VS Code extension** (interactive): open `notebooks/MTKD4SER_Google_Colab.ipynb`, then *Select Kernel → Colab →
  Auto Connect* with a T4. Colab Secrets (`userdata.get`) do not work in the extension yet, so the auth cells fall back
  to the `HF_TOKEN` / `WANDB_API_KEY` environment variables and then to a hidden password prompt.
- **`colab` CLI** (`pip install google-colab-cli`, scripted runs): `colab new -s mtkd --gpu T4`, `colab upload`,
  `colab exec -s mtkd -f script.py`, `colab stop -s mtkd`. To authenticate the CLI, run
  `gcloud auth application-default login` with the scopes listed in `colab skill`.

---

## Experiment Monitoring

```bash
tensorboard --logdir ./checkpoints/tb
```

---

## License

This project is licensed under the [MIT License](LICENSE).
