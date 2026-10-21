"""
models.py – model construction and checkpoint loading.

SERModel wraps any Hugging Face audio-classification backbone and:
  * applies the backbone's own input normalisation (zero-mean / unit-variance over
    the real, un-padded samples) on the GPU, so teachers and students with different
    preprocessing can share one prepared dataset;
  * always passes the attention mask, so padding is ignored by attention and by the
    mean-pooling in the classification head.

Because distillation only matches the 4 output logits, the student can be any size:
  --BACKBONE facebook/wav2vec2-base                  (95M, same as the teachers)
  --BACKBONE facebook/wav2vec2-base --NUM_LAYERS 4   (pretrained, first 4 transformer layers)
  --BACKBONE ntu-spml/distilhubert                   (24M, 2 transformer layers)
"""

import os

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from .checkpoints import load_checkpoint
from .constants import ID2LABEL, LABEL2ID, NUM_CLASSES


class SERModel(nn.Module):
    def __init__(self, hf_model, do_normalize: bool, backbone: str, num_layers: int | None):
        super().__init__()
        self.model = hf_model
        self.do_normalize = do_normalize
        self.backbone = backbone
        self.num_layers = num_layers

    @property
    def model_config(self) -> dict:
        """Everything needed to rebuild this model from a checkpoint."""
        return {"backbone": self.backbone, "num_layers": self.num_layers}

    @staticmethod
    def _normalize(x, mask):
        m = mask.to(x.dtype)
        n = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / n
        var = (((x - mean) * m) ** 2).sum(dim=1, keepdim=True) / n
        return ((x - mean) / torch.sqrt(var + 1e-7)) * m

    def forward(self, input_values, attention_mask):
        if self.do_normalize:
            input_values = self._normalize(input_values, attention_mask)
        return self.model(input_values, attention_mask=attention_mask).logits

    def num_parameters(self, trainable_only=False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def build_model(backbone: str, num_layers: int | None = None, device="cpu") -> SERModel:
    kwargs = {}
    if num_layers:
        kwargs["num_hidden_layers"] = int(num_layers)
    hf_model = AutoModelForAudioClassification.from_pretrained(
        backbone,
        num_labels=NUM_CLASSES,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
        **kwargs,
    )
    # Freeze the CNN feature encoder – standard practice for low-resource SER
    if hasattr(hf_model, "freeze_feature_encoder"):
        hf_model.freeze_feature_encoder()
    try:
        do_normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(backbone), "do_normalize", True))
    except Exception:
        do_normalize = True
    model = SERModel(hf_model, do_normalize, backbone, num_layers)
    return model.to(device)


def load_model_from_checkpoint(path: str, device="cpu", fallback_backbone="facebook/wav2vec2-base",
                               fallback_num_layers: int | None = None) -> tuple[SERModel, dict]:
    """
    Rebuild a model from a checkpoint. New checkpoints store their architecture under
    'model_config'; older ones don't, so fall back to the given backbone.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = load_checkpoint(path, device)
    cfg = ckpt.get("model_config") or {"backbone": fallback_backbone, "num_layers": fallback_num_layers}
    model = build_model(cfg["backbone"], cfg.get("num_layers"), device)
    model.model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def load_teacher(path: str, device="cpu", fallback_backbone="facebook/wav2vec2-base") -> SERModel:
    model, _ = load_model_from_checkpoint(path, device, fallback_backbone)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"[teacher] loaded {path} ({model.backbone}, layers={model.num_layers or 'all'})")
    return model
