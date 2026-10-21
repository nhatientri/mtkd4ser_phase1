# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-Teacher Knowledge Distillation for Speech Emotion Recognition (MTKD4SER), phase 1: English (IEMOCAP) and French (CaFE) teachers distilled into one multilingual student. 4 classes (anger, happiness, neutral, sadness). The main target is Google Colab Free Tier (T4, 15 GB); `notebooks/MTKD4SER_Google_Colab.ipynb` drives the same CLI. Library code is in the `mtkd4ser/` package, whose modules import each other with relative imports (`from .config import ...`). The entry points `main.py` and `summarize_results.py` stay at the repo root and import `from mtkd4ser.<module> import ...`. Keep them there, because the README, the notebook and `tests/test_smoke.py` run them as `python <script>.py`. The one-time data-prep scripts are in `data_prep/` and are run from the repo root as `python data_prep/<script>.py`. Their default paths (`data/`, and the raw `IEMOCAP_full_release` / `CaFE_48k` folders next to the repo) are relative to the repo root, not to `data_prep/`. There is no build or install step; scripts and tests rely on the repo root being on `sys.path` (`tests/conftest.py` adds it, and `data_prep/export_to_hf_dataset.py` inserts it itself). Module names below refer to files in `mtkd4ser/`.

## Commands

```bash
conda env create -f environment.yml && conda activate ser    # or: pip install -r requirements.txt
python -m pytest -q tests                                     # all tests (~1-2 min on CPU)
python -m pytest -q tests/test_distill.py                     # one file
python -m pytest -q tests/test_distill.py -k <name>           # one test
```

- `tests/test_smoke.py` is a fully offline end-to-end run. It builds a tiny random wav2vec2 and synthetic EN/FR Arrow datasets, then runs `main.py` as subprocesses for FT → KD → MTKD (every routing, plus a 1-layer student) → eval-only → `summarize_results.py`. If you change the CLI, the run naming or the result JSON, update this test.
- `tests/test_splits.py` checks for speaker leakage on the real splits under `data/` and is skipped when the data isn't there.
- No linter or formatter is configured. Pyrefly (`pyrefly.toml`) is set up for type checking against the `ser` conda env.

Running experiments (see README.md for the full workflow):
```bash
python main.py --PARADIGM FT --LANGUAGE EN --SESSION 1          # EN teacher
python main.py --PARADIGM FT --LANGUAGE FR --CAFE_FOLD 1        # FR teacher
python main.py --PARADIGM MTKD --LINGUALITY Multilingual --ROUTING language \
    --TEACHER_EN_CKPT <path|W&B artifact> --TEACHER_FR_CKPT <path|W&B artifact>
python main.py ... --TRAINING 0 --STUDENT_CKPT <path|W&B artifact>   # evaluate only
python summarize_results.py --SAVE_DIR checkpoints [--group]
```

Data prep (`data_prep/`): `reformat_datasets.py` (raw audio → 16 kHz, 4 classes, speaker-independent splits) → `export_to_hf_dataset.py` (CSV → Arrow) → `upload_to_hf.py` (push to the private HF Hub repo that Colab uses).

## Architecture

**Flow** (`main.py`): `parse_args` → `data.load_task_dataset` → `data.prepare_features` → for KD/MTKD, `pipeline.compute_teacher_logits` once per teacher, stacked to `(N, K, C)` → `distill.make_router` → `models.build_model` → `pipeline.Trainer(...).fit()`, which ends with `test()`.

**Config** (`config.py`): every field of the `Config` dataclass becomes an upper-case `--FLAG`, with allowed values in `CHOICES`. To add a flag, add a field (and a `CHOICES` entry if it has a fixed set of values). The full config is saved in every checkpoint and results JSON. `run_name()` builds names such as `MTKD_language_Multilingual_ENFR_S1F1_distilhubert_seed42` from paradigm, routing, data fold, backbone/depth and seed. Checkpoint paths, W&B artifact names and results files all use this name, so changing it breaks resolution of existing runs.

**One Trainer for every paradigm** (`pipeline.py`): FT is the loss with no KD term, and KD is MTKD with a single teacher routed `uniform`. Don't bring back separate per-paradigm loops (`changes/CHANGE5.md` explains why they were merged).

**Teacher logit cache**: teachers are frozen and see un-augmented audio, so their train-set logits are computed once, saved to `SAVE_DIR/teacher_cache/<sha1>.pt` (keyed on checkpoint path/size/mtime plus the dataset fingerprint and length), and the teachers are freed before training starts. The student's batches look up teacher logits through the `idx` column that `prepare_features` adds. Anything that reorders, filters or resamples the train split has to keep `idx` matching the row order used when the cache was built. Teachers don't run during validation or test.

**Distillation** (`distill.py`): KL is computed per (sample, teacher) and weighted per sample by router weights `w[B, K]`. The routers are `language` (an oracle that uses `language_id`), `uniform`, `cosine` (student vs teacher logits, softmax / `COSINE_TAU`) and `confidence` (1 − normalised entropy at T=1, softmax / `CONF_TAU`). The per-sample weighting is the main bug fix of CHANGE5: never reduce KL or weights over the batch before combining them.

**Padding and normalisation**: `prepare_features` stores raw, unpadded float32 waveforms truncated to `MAX_DURATION`. `utils.collate_fn` pads each batch and builds an attention mask. `models.SERModel` applies the backbone's own zero-mean/unit-variance normalisation on the device, over unpadded samples only. This lets one prepared dataset feed teachers and students with different backbones. Always pass `attention_mask` to the model.

**Models / checkpoints** (`models.py`, `checkpoints.py`, `tracking.py`): the student can be any HF audio-classification backbone, optionally truncated with `--NUM_LAYERS`, because only the 4 logits are distilled. `checkpoints.py` owns the checkpoint file format: `checkpoint_payload` builds the dict and `load_checkpoint` reads it. Every reader goes through `load_checkpoint`, so keep keys backward-compatible with existing `.pth` files. Checkpoints store `model_config` (backbone, num_layers) so they can be rebuilt; older teacher checkpoints without it fall back to `--TEACHER_BACKBONE`. `checkpoints.resolve_checkpoint` accepts a local path or a W&B artifact reference. The notebook imports it from `mtkd4ser.checkpoints`. `tracking.py` holds the W&B run helpers, which take `Config` and fail soft (they warn and training continues). `checkpoints` must not import `models` or `tracking`, to avoid import cycles. A run starts fresh unless `--RESUME 1` or `--RESUME_CKPT` is passed. Outputs: `SAVE_DIR/<run>.pth` (best), `<run>_last.pth`, `results/<run>.json`, `tb/`.

**Data** (`data.py`): for each dataset, the loaders try local Arrow (`data/hf_datasets/<name>/`), then the CSV splits from `reformat_datasets.py`, then the HF Hub (`--HF_DATASET_REPO`, `HF_TOKEN`). `MTKD_DATA_DIR` overrides `data/`. Splits are always speaker-independent: IEMOCAP is leave-one-session-out (`--SESSION 1-5`) and CaFE is leave-two-speakers-out (`--CAFE_FOLD 1-6`). Don't add a random dev split.

**Constants** (`constants.py`) is the single source for labels and language ids: `language_id` 0 = EN, 2 = FR, and 1 is reserved (Finnish was dropped). Multilingual runs pick the best epoch on `UR_lang_mean` (`--SELECT_METRIC auto`), and metrics are reported per language.

## Repo notes

- `changes/CHANGE1.md`–`CHANGE6.md` record design decisions and bugs that were fixed. Add a new `changes/CHANGE<N>.md` for each significant change. Read `changes/CHANGE5.md` before changing routing, padding, resume logic or run naming.
- `SAVE_DIR` defaults to `/content/checkpoints` when `/content` exists (Colab) and to `./checkpoints` otherwise. `data/` and checkpoints are gitignored.
