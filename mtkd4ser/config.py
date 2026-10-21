"""
config.py – one typed config object for every run (replaces the `hyperparam` dict).

The CLI flags are generated from the dataclass fields, so `--FIELD value` works for
every field below. The full config is saved inside every checkpoint and results file.
"""

import argparse
import os
from dataclasses import asdict, dataclass, fields

_DEFAULT_SAVE_DIR = "/content/checkpoints" if os.path.isdir("/content") else "./checkpoints"


@dataclass
class Config:
    # ---- what to train
    PARADIGM: str = "FT"                    # FT | KD | MTKD
    LANGUAGE: str = "EN"                    # EN | FR  (Monolingual only)
    LINGUALITY: str = "Monolingual"         # Monolingual | Multilingual (EN + FR)
    SESSION: int = 1                        # IEMOCAP leave-one-session-out fold (1-5)
    CAFE_FOLD: int = 1                      # CaFE leave-two-speakers-out fold (1-6)
    TRAINING: int = 1                       # 1 = train then test, 0 = test only

    # ---- model being trained (FT model, or KD/MTKD student)
    BACKBONE: str = "facebook/wav2vec2-base"
    NUM_LAYERS: int = 0                     # >0 keeps only the first N transformer layers (smaller student)
    TEACHER_BACKBONE: str = "facebook/wav2vec2-base"  # only used for old teacher checkpoints without model_config

    # ---- optimisation
    N_EPOCHS: int = 15
    LEARNING_RATE: float = 3e-5
    BATCH_SIZE: int = 8
    ACCUM_STEPS: int = 1                    # gradient accumulation (effective batch = BATCH_SIZE * ACCUM_STEPS)
    WARMUP_RATIO: float = 0.1               # linear warmup then linear decay
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 1.0
    AMP: int = 1                            # fp16 mixed precision on CUDA
    MAX_DURATION: float = 20.0              # seconds; longer clips are truncated
    NUM_WORKERS: int = 2
    BALANCE_LANG: int = 0                   # 1 = sample EN and FR equally often (multilingual)
    PATIENCE: int = 5
    SELECT_METRIC: str = "auto"             # auto = UR_lang_mean if dev has >1 language, else UR
    SEED: int = 42

    # ---- distillation
    ROUTING: str = "language"               # language | uniform | cosine | confidence
    LAMBDA_KD: float = 0.25
    TEMPERATURE: float = 5.0
    COSINE_TAU: float = 0.25
    CONF_TAU: float = 0.1
    TEACHER_CKPT: str | None = None         # KD (single teacher)
    TEACHER_EN_CKPT: str | None = None      # MTKD
    TEACHER_FR_CKPT: str | None = None      # MTKD

    # ---- checkpoints / IO
    SAVE_DIR: str = _DEFAULT_SAVE_DIR
    RUN_NAME: str | None = None             # default: built from the config (see run_name())
    STUDENT_CKPT: str | None = None         # checkpoint to evaluate when TRAINING=0
    RESUME: int = 0                         # 1 = resume from SAVE_DIR/<run>_last.pth if it exists
    RESUME_CKPT: str | None = None          # explicit path or W&B artifact to resume from
    SAVE_ALL_EPOCHS: int = 0
    HF_DATASET_REPO: str | None = None

    # ---- Weights & Biases
    USE_WANDB: int = 0
    WANDB_PROJECT: str = "mtkd4ser-phase1"
    WANDB_ENTITY: str | None = None
    WANDB_NAME: str | None = None
    WANDB_RUN_ID: str | None = None
    WANDB_LOG_MODEL: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


CHOICES = {
    "PARADIGM": ["FT", "KD", "MTKD"],
    "LANGUAGE": ["EN", "FR"],
    "LINGUALITY": ["Monolingual", "Multilingual"],
    "ROUTING": ["language", "uniform", "cosine", "confidence"],
    "SELECT_METRIC": ["auto", "UR", "UR_lang_mean", "WR"],
    "TRAINING": [0, 1], "AMP": [0, 1], "BALANCE_LANG": [0, 1], "RESUME": [0, 1],
    "SAVE_ALL_EPOCHS": [0, 1], "USE_WANDB": [0, 1], "WANDB_LOG_MODEL": [0, 1],
}


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="MTKD4SER Phase 1")
    for f in fields(Config):
        default = f.default
        if f.type in (int, "int"):
            typ = int
        elif f.type in (float, "float"):
            typ = float
        else:
            typ = str
        p.add_argument(f"--{f.name}", type=typ, default=default, choices=CHOICES.get(f.name))
    ns = p.parse_args(argv)
    return Config(**vars(ns))


def _short(backbone: str) -> str:
    return backbone.rstrip("/").split("/")[-1]


def run_name(cfg: Config) -> str:
    """
    e.g. FT_Monolingual_EN_S1_wav2vec2-base_seed42
         MTKD_language_Multilingual_ENFR_S1F1_distilhubert_seed42
    Includes backbone, depth and seed so runs with different settings never share a checkpoint.
    """
    if cfg.RUN_NAME:
        return cfg.RUN_NAME
    if cfg.LINGUALITY == "Multilingual":
        data_tag = f"ENFR_S{cfg.SESSION}F{cfg.CAFE_FOLD}"
    elif cfg.LANGUAGE == "EN":
        data_tag = f"EN_S{cfg.SESSION}"
    else:
        data_tag = f"FR_F{cfg.CAFE_FOLD}"
    model_tag = _short(cfg.BACKBONE) + (f"-L{cfg.NUM_LAYERS}" if cfg.NUM_LAYERS else "")
    parts = [cfg.PARADIGM]
    if cfg.PARADIGM == "MTKD":
        parts.append(cfg.ROUTING)
    parts += [cfg.LINGUALITY, data_tag, model_tag, f"seed{cfg.SEED}"]
    return "_".join(parts)
