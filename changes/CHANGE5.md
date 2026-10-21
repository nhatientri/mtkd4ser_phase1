# Change Summary (CHANGE 5): Correct per-sample routing, padding fix, cached teachers, smaller students

## 1. Bugs fixed (these change results – retrain teachers and re-run MTKD)

| Problem | Effect | Fix |
| :-- | :-- | :-- |
| KD loss = batch-mean KL × batch-mean routing weight | Every clip was distilled from every teacher in proportion to the batch's language mix (a French clip in a 7 EN / 1 FR batch got 87.5% of its signal from the EN teacher). On a 50/50 batch `language` routing was **identical** to `uniform`. | `distill.py`: KL computed per (sample, teacher) and weighted per sample. |
| `cosine` routing averaged over the batch | All clips in a batch got the same weights | Per-sample cosine (student vs teacher logits) |
| `confidence` routing used T=5-softened max-probabilities with tau=1 | Weights ≈ 0.51 / 0.49 – effectively uniform | 1 − normalised entropy of the T=1 prediction, sharpened with `CONF_TAU` (0.1) |
| `ds.map(..., padding=True)` padded each 1000-clip chunk to its longest clip (≈ 20 s), no attention mask | Average 4–5 s clips were ~75% zeros, and the classifier mean-pooled over the padding; ~4× wasted compute | Raw waveforms stored unpadded; padding per batch in `collate_fn` with an attention mask; normalisation done per model over real samples only (`models.SERModel`) |
| `--TRAINING 0` loaded `_last.pth` before the best checkpoint | Evaluated the last epoch, not the best | Evaluates `--STUDENT_CKPT` or the best checkpoint |
| Auto-resume of any existing `<run>_last.pth`; run name lacked backbone / seed | Re-running with new settings silently continued (or skipped) an old run; xls-r and base runs overwrote each other | Resume only with `--RESUME 1` / `--RESUME_CKPT`; run name includes backbone, depth, seed |
| Notebook passed `--STUDENT_CKPT` (not defined) | Evaluation cell crashed | Flag added |
| `epochs_no_improve` not restored on resume; WR printed as UR | – | Fixed |

## 2. Evaluation

* Dev/test metrics per language (`UR_en`, `UR_fr`, `UR_lang_mean`) and per-language confusion matrices.
* Multilingual runs select the best epoch on `UR_lang_mean` (dev was 481 EN clips from one speaker vs 84 FR).
* `--BALANCE_LANG 1`: language-balanced sampling (FR is ~8% of training clips).
* `--CAFE_FOLD 1-6` now works (was ignored); `export_to_hf_dataset.py --all_cafe_folds` / `upload_to_hf.py --all_cafe_folds`.
* The old raw-JSON fallback (random, speaker-leaking dev split) was removed; loaders use only the speaker-independent splits from `reformat_datasets.py`.

## 3. Efficiency

* Teacher logits computed once per (teacher checkpoint, dataset) and cached in `SAVE_DIR/teacher_cache/`; teachers leave GPU memory before training. Teachers are no longer run during validation/test.
* fp16 mixed precision (`--AMP 1`), linear warmup + decay (`--WARMUP_RATIO`), gradient clipping, gradient accumulation (`--ACCUM_STEPS`).
* `SAVE_ALL_EPOCHS` now defaults to 0 (each checkpoint with optimizer state is ~1.1 GB for wav2vec2-base).
* Every run writes `SAVE_DIR/results/<run>.json` with accuracy **and** efficiency (params, s/epoch, samples/s, peak GPU memory, teacher-cache time, test ms/clip). `summarize_results.py [--group]` tabulates them.

## 4. Smaller students

Distillation only matches the 4 logits, so the student no longer has to share the teachers' architecture:
`--BACKBONE ntu-spml/distilhubert` (~24M), `--BACKBONE facebook/wav2vec2-base --NUM_LAYERS 4` (~38M) or `6` (~52M).
Teacher architecture is read from each checkpoint's `model_config` (older checkpoints fall back to `--TEACHER_BACKBONE`).

## 5. Code structure

* One `Trainer` (`pipeline.py`) replaces `pipeline_ft` / `pipeline_kd` / `pipeline_mtkd` and the six `train_*`/`eval_*` loops. KD is MTKD with one teacher; FT is the loss without the KD term.
* `config.py`: typed `Config` dataclass (every field is a `--FLAG`), saved in every checkpoint and results file.
* `constants.py`: labels and language ids in one place. FESC / Finnish code removed for Phase 1 (`--INCLUDE_FI`, `--TEACHER_FI_CKPT` are gone).
* W&B helpers moved unchanged to `checkpoints.py` (`from pipeline import resolve_checkpoint` still works).
* Global `warnings.filterwarnings("ignore")` removed.
* `tests/`: routing/loss, batching/normalisation/metrics, speaker-leakage on the real splits, and an offline end-to-end smoke run with a tiny random wav2vec2.

## 6. CLI changes

* New: `--CAFE_FOLD`, `--NUM_LAYERS`, `--TEACHER_BACKBONE`, `--STUDENT_CKPT`, `--RESUME`, `--BALANCE_LANG`, `--CONF_TAU`, `--AMP`, `--ACCUM_STEPS`, `--WARMUP_RATIO`, `--WEIGHT_DECAY`, `--MAX_GRAD_NORM`, `--MAX_DURATION`, `--NUM_WORKERS`, `--SELECT_METRIC`, `--RUN_NAME`.
* Removed: `--INCLUDE_FI`, `--TEACHER_FI_CKPT`.
* Run / artifact names changed, e.g. `FT_Monolingual_EN_S1_wav2vec2-base_seed42` (W&B artifact `…_seed42-best:best`).
