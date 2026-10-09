# Change Summary (CHANGE 3): Dynamic Bilingual & Multilingual MTKD

This document summarizes the optimization performed to remove the Finnish (FI) dependency and enable dynamic teacher configurations (specifically **English + French** only) in the **MTKD4SER Phase 1** codebase.

---

## 1. Problem & Bottlenecks in Previous Implementation

When training without the Finnish dataset (FESC), the original code exhibited three critical bottlenecks:

1. **GPU VRAM Waste (Memory Bottleneck)**:
   - `model_mtkd` in `models.py` hardcoded loading 3 teachers (`en`, `fi`, `fr`).
   - When Finnish was omitted, `main.py` duplicated the English teacher checkpoint as a dummy Finnish teacher model.
   - This loaded 4 complete models into VRAM (1 student + 3 teachers) instead of 3, wasting ~380 MB of VRAM for `wav2vec2-base` and **~1.27 GB for `xls-r-300m`**, increasing CUDA Out-Of-Memory (OOM) risk on Google Colab T4 (15 GB).

2. **Redundant Inference (Compute Bottleneck)**:
   - For every batch in `train_mtkd` and `eval_mtkd`, inference was executed on all teachers: `t_logits = {k: m(x).logits for k, m in teachers.items()}`.
   - The dummy Finnish teacher ran an entire forward pass (CNN + 12 Transformer layers) on the exact same audio batch `x`, resulting in **~33% wasted computation time** and redundant KL divergence calculations per batch.

3. **Experimental Bias in Routing Ablations (Algorithmic Flaw)**:
   - **Uniform routing**: Fixed at `1/3` across 3 slots. Because FI was a duplicate of EN, English effectively received **$2/3$ (66.7%)** weight and French only **$1/3$ (33.3%)**, destroying the 50/50 baseline.
   - **Cosine & Confidence routing**: Softmax was calculated across 3 slots. The duplicate English teacher had identical logits to EN, receiving a second slot in the competition and unfairly suppressing the French teacher's probability.
   - **Language routing**: Mathematically correct (FI weight was 0), but still suffered the full memory and forward compute penalty.

---

## 2. Changes Made & Implementation Details

### A. Dynamic Model Loading ([models.py](file:///Users/hunter/Downloads/mtkd4ser_phase1/models.py))
- **Refactored `model_mtkd`**:
  - Dynamically builds `teacher_ckpts` using only non-null checkpoints provided by the user (`teacher_en_ckpt`, `teacher_fr_ckpt`, `teacher_fi_ckpt`).
  - Validates that at least 2 teachers are provided (`len(teacher_ckpts) >= 2`) without mandating a fixed set of 3 languages.
  - Instantiates and loads **only the active teachers into GPU memory**.
  - When running EN + FR, only 2 teachers are created (saving 1 full backbone from VRAM).

### B. Dynamic Routing & Distillation ([utils.py](file:///Users/hunter/Downloads/mtkd4ser_phase1/utils.py))
- **Added `_compute_routing_weights()`**:
  - Dynamically produces weights of shape `(B, num_teachers)` based on the active language keys (`lang_keys = list(teachers.keys())`).
  - **`uniform`**: Set to `1.0 / num_teachers` (exactly $0.5$ for EN and $0.5$ for FR).
  - **`cosine` & `confidence`**: Evaluates cosine similarity and confidence max softmax strictly over the active teachers ($N=2$).
  - **`language` (Oracle)**: Vectorized lookup tensor mapping dataset `language_id` (`0` for EN, `2` for FR) directly to the corresponding active teacher column index (`0` for EN, `1` for FR).
- **Refactored `train_mtkd()` & `eval_mtkd()`**:
  - Forward passes run **only on active teachers** (`t_logits = {k: teachers[k](x).logits for k in lang_keys}`).
  - KL divergence is computed only for active teachers and weighted by the dynamic routing matrix.
  - Diagnostic logging dynamically outputs weights for active languages (e.g. `[routing=language] avg weights EN=0.922 FR=0.078`).

### C. Entry Point Optimization ([main.py](file:///Users/hunter/Downloads/mtkd4ser_phase1/main.py))
- Removed the fallback: `teacher_fi_ckpt = args.TEACHER_FI_CKPT or args.TEACHER_EN_CKPT`.
- Passed `teacher_fi_ckpt=args.TEACHER_FI_CKPT` directly (which defaults to `None`).
- Simplified `session_en=args.SESSION` in multilingual data building to correctly reflect the requested fold.

### D. Dataset Session Guard ([data.py](file:///Users/hunter/Downloads/mtkd4ser_phase1/data.py))
- In `build_multilingual()`, guarded loading of `data/hf_datasets/combined` with `session_en == 1 and session_fr == 1`.
- If another fold is requested (e.g. Session 2), it properly loads `iemocap(session=2)` and `cafe(session=1)` instead of reusing Session 1's combined split.

### E. Documentation & Batch Size Advice ([README_PHASE1.md](file:///Users/hunter/Downloads/mtkd4ser_phase1/README_PHASE1.md))
- Updated batch size recommendation table for Google Colab Free (T4 15 GB):
  - `wav2vec2-base`: 6–12 batch size with 2 teachers (vs 4–8 with 3 teachers).
  - `xls-r-300m`: 4–6 batch size with 2 teachers (vs 2–4 with 3 teachers).

---

## 3. Verification & Testing

1. **Routing Weights Unit Test**:
   - Verified `_compute_routing_weights` on synthetic batches for:
     - **2 Teachers (`EN`, `FR`)**:
       - `uniform` $\rightarrow$ `[[0.5000, 0.5000], ...]`
       - `cosine` $\rightarrow$ shape `(4, 2)`, sum to 1.0 per row.
       - `confidence` $\rightarrow$ shape `(4, 2)`, sum to 1.0 per row.
       - `language` $\rightarrow$ sample `lang_id=0` (EN) gave `[1.0, 0.0]`; `lang_id=2` (FR) gave `[0.0, 1.0]`.
     - **3 Teachers (`EN`, `FI`, `FR`)**: Verified full backward-compatibility ($1/3$ uniform, 3-way softmax, and `0/1/2` one-hot).
2. **CLI & Argument Parsing**:
   - Verified `python main.py --help` runs cleanly without syntax or import errors.

---

## 4. How to Run (2 Teachers: EN + FR)

```bash
# Multi-Teacher Distillation (Oracle Language Routing)
python main.py --PARADIGM MTKD --LANGUAGE EN --LINGUALITY Multilingual \
  --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 6 --LEARNING_RATE 3e-5 \
  --TRAINING 1 --ROUTING language \
  --TEACHER_EN_CKPT /content/checkpoints/FT_Monolingual_EN_S1.pth \
  --TEACHER_FR_CKPT /content/checkpoints/FT_Monolingual_FR_S1.pth \
  --INCLUDE_FI 0 --SAVE_DIR /content/checkpoints

# Routing Ablations
python main.py ... --ROUTING uniform ...
python main.py ... --ROUTING cosine ...
python main.py ... --ROUTING confidence ...
```
