"""
main.py – Phase 1 entry point (Colab Free friendly)

Example commands
----------------
# 1. Fine-tune English monolingual teacher
python main.py --PARADIGM FT --LANGUAGE EN --LINGUALITY Monolingual \
    --SESSION 1 --N_EPOCHS 15 --BATCH_SIZE 8 --LEARNING_RATE 3e-5 --TRAINING 1

# 2. Fine-tune French monolingual teacher
python main.py --PARADIGM FT --LANGUAGE FR --LINGUALITY Monolingual \
    --SESSION 1 --N_EPOCHS 15 --BATCH_SIZE 8 --LEARNING_RATE 3e-5 --TRAINING 1

# 3. MTKD with oracle language routing (recommended)
python main.py --PARADIGM MTKD --LANGUAGE EN --LINGUALITY Multilingual \
    --SESSION 1 --N_EPOCHS 15 --BATCH_SIZE 4 --LEARNING_RATE 3e-5 --TRAINING 1 \
    --ROUTING language \
    --TEACHER_EN_CKPT /content/checkpoints/FT_Monolingual_EN_S1.pth \
    --TEACHER_FR_CKPT /content/checkpoints/FT_Monolingual_FR_S1.pth

# 4. Ablation: cosine (original paper) vs uniform vs language
python main.py ... --ROUTING cosine
python main.py ... --ROUTING uniform
"""

import os
import random
import argparse
import numpy as np
from typing import cast
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor

from data import iemocap, cafe, fesc, build_multilingual
from models import model_ft, model_kd, model_mtkd
from pipeline import pipeline_ft, pipeline_kd, pipeline_mtkd
from utils import preprocess_function, collate_fn
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="MTKD4SER Phase-1")
    p.add_argument("--N_EPOCHS", type=int, default=15)
    p.add_argument("--LEARNING_RATE", type=float, default=3e-5)
    p.add_argument("--BATCH_SIZE", type=int, default=8)
    p.add_argument("--SESSION", type=int, default=1)
    p.add_argument("--TRAINING", type=int, default=1, choices=[0, 1])
    p.add_argument("--PARADIGM", type=str, default="FT",
                   choices=["FT", "KD", "MTKD"])
    p.add_argument("--LANGUAGE", type=str, default="EN",
                   choices=["EN", "FI", "FR"])
    p.add_argument("--LINGUALITY", type=str, default="Monolingual",
                   choices=["Monolingual", "Multilingual"])
    p.add_argument("--BACKBONE", type=str, default="facebook/wav2vec2-base",
                   help="e.g. facebook/wav2vec2-base | facebook/wav2vec2-xls-r-300m")
    p.add_argument("--ROUTING", type=str, default="language",
                   choices=["language", "uniform", "cosine", "confidence"])
    p.add_argument("--LAMBDA_KD", type=float, default=0.25)
    p.add_argument("--TEMPERATURE", type=float, default=5.0)
    p.add_argument("--COSINE_TAU", type=float, default=0.25)
    p.add_argument("--PATIENCE", type=int, default=5,
                   help="Early stopping patience (epochs without validation UR improvement)")
    default_save_dir = "/content/checkpoints" if os.path.isdir("/content") else "./checkpoints"
    p.add_argument("--SAVE_DIR", type=str, default=default_save_dir)
    p.add_argument("--SEED", type=int, default=42)
    # Teacher checkpoints for KD / MTKD
    p.add_argument("--TEACHER_CKPT", type=str, default=None,
                   help="Single teacher for KD")
    p.add_argument("--TEACHER_EN_CKPT", type=str, default=None)
    p.add_argument("--TEACHER_FI_CKPT", type=str, default=None)
    p.add_argument("--TEACHER_FR_CKPT", type=str, default=None)
    p.add_argument("--INCLUDE_FI", type=int, default=0,
                   help="1 = try to load FESC in multilingual mode")
    p.add_argument("--HF_DATASET_REPO", type=str, default=None,
                   help="Optional Hugging Face dataset repository (e.g. username/mtkd4ser-phase1)")
    # Weights & Biases experiment tracking
    p.add_argument("--USE_WANDB", type=int, default=0, choices=[0, 1],
                   help="1 = enable Weights & Biases experiment tracking")
    p.add_argument("--WANDB_PROJECT", type=str, default="mtkd4ser-phase1",
                   help="Weights & Biases project name")
    p.add_argument("--WANDB_ENTITY", type=str, default=None,
                   help="Weights & Biases team/entity name")
    p.add_argument("--WANDB_NAME", type=str, default=None,
                   help="Custom run name for Weights & Biases")
    p.add_argument("--WANDB_RUN_ID", type=str, default=None,
                   help="Existing Weights & Biases Run ID to resume logging into the same dashboard run")
    p.add_argument("--WANDB_LOG_MODEL", type=int, default=1, choices=[0, 1],
                   help="1 = upload best checkpoint as a W&B Artifact")
    p.add_argument("--RESUME_CKPT", type=str, default=None,
                   help="Path or W&B Artifact URI (e.g. 'entity/project/name:latest') to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.SEED)

    if args.HF_DATASET_REPO:
        os.environ["HF_DATASET_REPO"] = args.HF_DATASET_REPO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.BACKBONE} | Routing: {args.ROUTING}")

    # ------------------------------------------------------------------ data
    if args.LINGUALITY == "Monolingual":
        if args.LANGUAGE == "EN":
            label2id, id2label, ds = iemocap(session=args.SESSION, hf_dataset_repo=args.HF_DATASET_REPO)
        elif args.LANGUAGE == "FR":
            label2id, id2label, ds = cafe(session=args.SESSION, hf_dataset_repo=args.HF_DATASET_REPO)
        else:
            label2id, id2label, ds = fesc(session=args.SESSION)
    else:
        label2id, id2label, ds = build_multilingual(
            session_en=args.SESSION,
            session_fi=9,
            session_fr=1,
            include_fi=bool(args.INCLUDE_FI),
            hf_dataset_repo=args.HF_DATASET_REPO,
        )

    print(ds)
    print("label2id:", label2id)

    # ------------------------------------------------------------------ feature extractor
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.BACKBONE)
    except Exception:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.BACKBONE)

    encoded = ds.map(
        lambda ex: preprocess_function(ex, feature_extractor),
        remove_columns=["audio"],
        batched=True,
    )

    train_loader = DataLoader(
        cast(TorchDataset, encoded["train"]), batch_size=args.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(
        cast(TorchDataset, encoded["dev"]), batch_size=args.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(
        cast(TorchDataset, encoded["test"]), batch_size=args.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2)

    num_classes = len(label2id)
    label_names = [id2label[i] for i in range(num_classes)]

    hyperparam = {
        "PARADIGM": args.PARADIGM,
        "SESSION": args.SESSION,
        "LEARNING_RATE": args.LEARNING_RATE,
        "BATCH_SIZE": args.BATCH_SIZE,
        "TRAINING": args.TRAINING,
        "N_EPOCHS": args.N_EPOCHS,
        "LINGUALITY": args.LINGUALITY,
        "LANGUAGE": args.LANGUAGE,
        "BACKBONE": args.BACKBONE,
        "SAVE_DIR": args.SAVE_DIR,
        "PATIENCE": args.PATIENCE,
        "TEMPERATURE": args.TEMPERATURE,
        "LAMBDA_KD": args.LAMBDA_KD,
        "ROUTING": args.ROUTING,
        "COSINE_TAU": args.COSINE_TAU,
        "USE_WANDB": args.USE_WANDB,
        "WANDB_PROJECT": args.WANDB_PROJECT,
        "WANDB_ENTITY": args.WANDB_ENTITY,
        "WANDB_NAME": args.WANDB_NAME,
        "WANDB_RUN_ID": args.WANDB_RUN_ID,
        "WANDB_LOG_MODEL": args.WANDB_LOG_MODEL,
        "RESUME_CKPT": args.RESUME_CKPT,
        "label_names": label_names,
        "TEACHER_CKPT": args.TEACHER_CKPT,
        "TEACHER_EN_CKPT": args.TEACHER_EN_CKPT,
        "TEACHER_FI_CKPT": args.TEACHER_FI_CKPT,
        "TEACHER_FR_CKPT": args.TEACHER_FR_CKPT,
    }

    # ------------------------------------------------------------------ models & run
    from pipeline import resolve_checkpoint

    teacher_ckpt = resolve_checkpoint(args.TEACHER_CKPT, save_dir=args.SAVE_DIR,
                                      default_project=args.WANDB_PROJECT, default_entity=args.WANDB_ENTITY)
    teacher_en_ckpt = resolve_checkpoint(args.TEACHER_EN_CKPT, save_dir=args.SAVE_DIR,
                                         default_project=args.WANDB_PROJECT, default_entity=args.WANDB_ENTITY)
    teacher_fi_ckpt = resolve_checkpoint(args.TEACHER_FI_CKPT, save_dir=args.SAVE_DIR,
                                         default_project=args.WANDB_PROJECT, default_entity=args.WANDB_ENTITY)
    teacher_fr_ckpt = resolve_checkpoint(args.TEACHER_FR_CKPT, save_dir=args.SAVE_DIR,
                                         default_project=args.WANDB_PROJECT, default_entity=args.WANDB_ENTITY)

    if args.PARADIGM == "FT":
        model = model_ft(label2id, id2label, num_classes, device, args.BACKBONE)
        pipeline_ft(model, train_loader, valid_loader, test_loader,
                    hyperparam, device)

    elif args.PARADIGM == "KD":
        teacher, student = model_kd(
            label2id, id2label, num_classes, device,
            backbone=args.BACKBONE,
            teacher_ckpt=teacher_ckpt,
        )
        pipeline_kd(teacher, student, train_loader, valid_loader, test_loader,
                    hyperparam, device)

    else:  # MTKD
        teachers, student = model_mtkd(
            label2id, id2label, num_classes, device,
            backbone=args.BACKBONE,
            teacher_en_ckpt=teacher_en_ckpt,
            teacher_fi_ckpt=teacher_fi_ckpt,
            teacher_fr_ckpt=teacher_fr_ckpt,
        )
        pipeline_mtkd(teachers, student, train_loader, valid_loader, test_loader,
                      hyperparam, device)


if __name__ == "__main__":
    main()
