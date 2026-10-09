"""
models.py – Phase 1
- Flexible backbone
- Safe teacher loading (explicit error if missing)
- Teachers always frozen + eval-ready
"""

import os
import torch
from transformers import AutoModelForAudioClassification
import warnings
warnings.filterwarnings("ignore")


def _load_backbone(backbone_name, label2id, id2label, num_classes, device):
    model = AutoModelForAudioClassification.from_pretrained(
        backbone_name,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    # Freeze CNN feature encoder – standard practice for low-resource SER
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    model.to(device)
    return model


def model_ft(label2id, id2label, num_classes=4, device="cpu",
             backbone="facebook/wav2vec2-base"):
    """Vanilla fine-tuning student / model."""
    return _load_backbone(backbone, label2id, id2label, num_classes, device)


def model_kd(label2id, id2label, num_classes=4, device="cpu",
             backbone="facebook/wav2vec2-base",
             teacher_ckpt=None):
    """
    Single-teacher KD.
    teacher_ckpt: path to a previously trained FT checkpoint (.pth).
                  If None or file missing → raise clear error.
    """
    teacher = _load_backbone(backbone, label2id, id2label, num_classes, device)
    student = _load_backbone(backbone, label2id, id2label, num_classes, device)

    if teacher_ckpt is None or not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError(
            f"[KD] Teacher checkpoint not found: {teacher_ckpt}\n"
            "Train a monolingual/multilingual FT model first and pass its path."
        )

    ckpt = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    print(f"[KD] Loaded teacher from {teacher_ckpt}")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    return teacher, student


def model_mtkd(label2id, id2label, num_classes=4, device="cpu",
               backbone="facebook/wav2vec2-base",
               teacher_en_ckpt=None,
               teacher_fi_ckpt=None,
               teacher_fr_ckpt=None,
               teacher_ckpts=None):
    """
    Multi-teacher setup supporting dynamic subsets (e.g., EN+FR only or EN+FI+FR).
    Only loads checkpoints that are explicitly specified (no dummy duplicate models).
    Returns dict of loaded teachers + student.
    """
    if teacher_ckpts is None:
        teacher_ckpts = {}
        if teacher_en_ckpt:
            teacher_ckpts["en"] = teacher_en_ckpt
        if teacher_fi_ckpt:
            teacher_ckpts["fi"] = teacher_fi_ckpt
        if teacher_fr_ckpt:
            teacher_ckpts["fr"] = teacher_fr_ckpt

    if len(teacher_ckpts) < 2:
        raise ValueError(
            f"[MTKD] Multi-teacher distillation requires at least 2 teachers, "
            f"but found {len(teacher_ckpts)}: {list(teacher_ckpts.keys())}.\n"
            "Provide checkpoints for at least 2 languages (e.g. --TEACHER_EN_CKPT and --TEACHER_FR_CKPT)."
        )

    teachers = {}
    for lang, path in teacher_ckpts.items():
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f"[MTKD] Teacher checkpoint for '{lang}' not found: {path}\n"
                f"Please ensure the {lang.upper()} teacher checkpoint exists."
            )
        model = _load_backbone(backbone, label2id, id2label, num_classes, device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        teachers[lang] = model
        print(f"[MTKD] Loaded {lang.upper()} teacher from {path}")

    student = _load_backbone(backbone, label2id, id2label, num_classes, device)
    print(f"[MTKD] Configured {len(teachers)} active teachers: {list(teachers.keys())}")
    return teachers, student

