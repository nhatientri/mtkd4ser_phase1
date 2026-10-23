"""
main.py – MTKD4SER Phase 1 entry point.

Examples
--------
# 1. Teachers (full wav2vec2-base)
python main.py --PARADIGM FT --LANGUAGE EN --SESSION 1
python main.py --PARADIGM FT --LANGUAGE FR --CAFE_FOLD 1

# 2. Baseline every MTKD result must beat: multilingual fine-tuning, no teachers
python main.py --PARADIGM FT --LINGUALITY Multilingual --SESSION 1 --CAFE_FOLD 1

# 3. MTKD student (same size as the teachers)
python main.py --PARADIGM MTKD --LINGUALITY Multilingual --ROUTING language \
    --TEACHER_EN_CKPT checkpoints/FT_Monolingual_EN_S1_wav2vec2-base_seed42.pth \
    --TEACHER_FR_CKPT checkpoints/FT_Monolingual_FR_F1_wav2vec2-base_seed42.pth

# 4. Smaller, faster students (teacher logits are cached, so only the student trains)
python main.py ... --BACKBONE facebook/wav2vec2-base --NUM_LAYERS 4
python main.py ... --BACKBONE ntu-spml/distilhubert

# 5. Evaluate a saved model
python main.py --PARADIGM MTKD --LINGUALITY Multilingual --TRAINING 0 --STUDENT_CKPT <path or W&B artifact>

# 6. Compare runs
python summarize_results.py --SAVE_DIR checkpoints
"""

import os
import random
import warnings

import numpy as np
import torch

from mtkd4ser.checkpoints import checkpoint_config, resolve_checkpoint
from mtkd4ser.config import Config, parse_args, run_name
from mtkd4ser.data import load_task_dataset, prepare_features
from mtkd4ser.distill import make_router
from mtkd4ser.models import build_model, load_model_from_checkpoint
from mtkd4ser.pipeline import Trainer, compute_teacher_logits

# Silence only known-noisy third-party warnings (not everything)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve(cfg: Config, ref):
    return resolve_checkpoint(ref, save_dir=cfg.SAVE_DIR, default_project=cfg.WANDB_PROJECT,
                              default_entity=cfg.WANDB_ENTITY)


def teacher_checkpoints(cfg: Config) -> dict[str, str]:
    if cfg.PARADIGM == "KD":
        refs = {"teacher": cfg.TEACHER_CKPT}
    elif cfg.PARADIGM == "MTKD":
        refs = {"en": cfg.TEACHER_EN_CKPT, "fr": cfg.TEACHER_FR_CKPT}
    else:
        return {}
    out = {}
    for key, ref in refs.items():
        path = _resolve(cfg, ref) if ref else None
        if not path or not os.path.isfile(path):
            flag = "--TEACHER_CKPT" if key == "teacher" else f"--TEACHER_{key.upper()}_CKPT"
            raise FileNotFoundError(f"[{cfg.PARADIGM}] teacher checkpoint for '{key}' not found: {ref!r} ({flag})")
        _check_teacher_folds(cfg, key, path)
        out[key] = path
    return out


def _data_folds(c: dict) -> dict:
    """The fold fields that decide which speakers a run trains on."""
    multi = c.get("LINGUALITY") == "Multilingual"
    folds = {}
    if multi or c.get("LANGUAGE") == "EN":
        folds["SESSION"] = c.get("SESSION")
    if multi or c.get("LANGUAGE") == "FR":
        folds["CAFE_FOLD"] = c.get("CAFE_FOLD")
    return folds


def _check_teacher_folds(cfg: Config, key: str, path: str):
    """A teacher trained on another fold has seen this run's dev/test speakers (leakage)."""
    t_cfg = checkpoint_config(path)
    if not t_cfg:
        print(f"[warn] teacher '{key}' ({path}) stores no config – cannot check that it uses the same folds.")
        return
    mine = _data_folds(cfg.to_dict())
    for field, value in _data_folds(t_cfg).items():
        if field in mine and mine[field] != value:
            raise ValueError(f"[{cfg.PARADIGM}] teacher '{key}' ({path}) was trained with {field}={value}, but this "
                             f"run uses {field}={mine[field]}: its training data contains this run's dev/test "
                             f"speakers. Use a teacher trained on {field}={mine[field]}.")


def main():
    cfg = parse_args()
    set_seed(cfg.SEED)
    if cfg.HF_DATASET_REPO:
        os.environ["HF_DATASET_REPO"] = cfg.HF_DATASET_REPO
    hf_repo = cfg.HF_DATASET_REPO or os.environ.get("HF_DATASET_REPO")

    device = pick_device()
    print(f"Device: {device} | run: {run_name(cfg)}")

    # teachers are resolved and fold-checked before any audio is decoded, so mistakes fail fast
    t_ckpts = teacher_checkpoints(cfg) if cfg.TRAINING else {}

    # ------------------------------------------------------------------ data
    raw = load_task_dataset(cfg.LINGUALITY, cfg.LANGUAGE, cfg.SESSION, cfg.CAFE_FOLD, hf_repo)
    print(raw)
    splits = prepare_features(raw, max_duration=cfg.MAX_DURATION)

    # ------------------------------------------------------------------ evaluation only
    if not cfg.TRAINING:
        ref = cfg.STUDENT_CKPT or os.path.join(cfg.SAVE_DIR, f"{run_name(cfg)}.pth")
        path = _resolve(cfg, ref)
        model, _ = load_model_from_checkpoint(path, device, fallback_backbone=cfg.BACKBONE,
                                              fallback_num_layers=cfg.NUM_LAYERS or None)
        Trainer(cfg, model, splits, device).test(ckpt_path=path)
        return

    # ------------------------------------------------------------------ teachers → cached logits
    teacher_logits, router, cache_sec = None, None, 0.0
    if t_ckpts:
        per_teacher = []
        for path in t_ckpts.values():
            logits, secs = compute_teacher_logits(path, splits["train"], device, cfg)
            per_teacher.append(logits)
            cache_sec += secs
        teacher_logits = torch.stack(per_teacher, dim=1)                       # (N, K, C)
        langs = list(t_ckpts.keys())
        mode = cfg.ROUTING if cfg.PARADIGM == "MTKD" else "uniform"           # KD = 1 teacher
        router = make_router(mode, langs, cfg.COSINE_TAU, cfg.CONF_TAU)

    # ------------------------------------------------------------------ student / model
    model = build_model(cfg.BACKBONE, cfg.NUM_LAYERS or None, device)
    Trainer(cfg, model, splits, device, teacher_logits, router, cache_sec).fit()


if __name__ == "__main__":
    main()
