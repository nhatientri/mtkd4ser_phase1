"""
End-to-end smoke test, fully offline (~1-2 min on CPU):
a tiny random wav2vec2 + synthetic EN/FR audio → FT teachers → KD → MTKD (all routings,
including a smaller 1-layer student) → evaluation-only → summarize_results.

    pytest -q tests/test_smoke.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_tiny_backbone(path):
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    cfg = Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        conv_dim=(32, 32, 32), conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        classifier_proj_size=16, num_labels=4, mask_time_prob=0.0, layerdrop=0.0,
    )
    Wav2Vec2ForSequenceClassification(cfg).save_pretrained(path)
    Wav2Vec2FeatureExtractor(do_normalize=True, return_attention_mask=True).save_pretrained(path)


def _make_data(data_dir):
    """Synthetic Arrow datasets with the same layout export_to_hf_dataset.py produces."""
    import soundfile as sf
    from datasets import Audio, Dataset, DatasetDict

    rng = np.random.default_rng(0)
    wav_dir = os.path.join(data_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    def split(name, n, lang, speaker):
        paths, labels = [], []
        for i in range(n):
            label = i % 4
            dur = rng.uniform(0.4, 1.5)
            t = np.arange(int(16000 * dur)) / 16000
            x = 0.3 * np.sin(2 * np.pi * (150 + 100 * label) * t) + 0.05 * rng.standard_normal(len(t))
            p = os.path.join(wav_dir, f"{name}_{lang}_{i}.wav")
            sf.write(p, x.astype(np.float32), 16000)
            paths.append(p)
            labels.append(label)
        d = Dataset.from_dict({"audio": paths, "label": labels, "language_id": [lang] * n,
                               "speaker_id": [speaker] * n})
        return d.cast_column("audio", Audio(sampling_rate=16000))

    hf = os.path.join(data_dir, "hf_datasets")
    DatasetDict({"train": split("tr", 24, 0, "en_a"), "dev": split("dv", 8, 0, "en_b"),
                 "test": split("te", 8, 0, "en_c")}).save_to_disk(os.path.join(hf, "iemocap_session1"))
    DatasetDict({"train": split("tr", 16, 2, "fr_a"), "dev": split("dv", 8, 2, "fr_b"),
                 "test": split("te", 8, 2, "fr_c")}).save_to_disk(os.path.join(hf, "cafe"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("soundfile")
    root = tmp_path_factory.mktemp("smoke")
    backbone = str(root / "tiny-w2v2")
    _make_tiny_backbone(backbone)
    _make_data(str(root / "data"))
    e = dict(os.environ, MTKD_DATA_DIR=str(root / "data"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
             CUDA_VISIBLE_DEVICES="")
    return {"root": root, "backbone": backbone, "save": str(root / "ckpt"), "env": e}


def _run(env, *args):
    cmd = [sys.executable, os.path.join(PROJECT, "main.py"), "--BACKBONE", env["backbone"],
           "--SAVE_DIR", env["save"], "--N_EPOCHS", "2", "--BATCH_SIZE", "4", "--NUM_WORKERS", "0",
           "--LEARNING_RATE", "1e-3", *args]
    r = subprocess.run(cmd, env=env["env"], cwd=PROJECT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + "\n" + r.stderr[-5000:]
    return r.stdout


def _results(env, name):
    with open(os.path.join(env["save"], "results", f"{name}.json")) as f:
        return json.load(f)


def test_pipeline_end_to_end(env):
    tag = os.path.basename(env["backbone"])
    _run(env, "--PARADIGM", "FT", "--LANGUAGE", "EN")
    _run(env, "--PARADIGM", "FT", "--LANGUAGE", "FR")
    en = os.path.join(env["save"], f"FT_Monolingual_EN_S1_{tag}_seed42.pth")
    fr = os.path.join(env["save"], f"FT_Monolingual_FR_F1_{tag}_seed42.pth")
    assert os.path.isfile(en) and os.path.isfile(fr)

    # multilingual FT baseline – per-language metrics present
    _run(env, "--PARADIGM", "FT", "--LINGUALITY", "Multilingual")
    r = _results(env, f"FT_Multilingual_ENFR_S1F1_{tag}_seed42")
    assert {"UR_en", "UR_fr", "UR_lang_mean"} <= set(r["test"])

    # single-teacher KD
    _run(env, "--PARADIGM", "KD", "--LINGUALITY", "Multilingual", "--TEACHER_CKPT", en)

    # MTKD, every routing, smaller (1-layer) student
    for routing in ("language", "uniform", "cosine", "confidence"):
        out = _run(env, "--PARADIGM", "MTKD", "--LINGUALITY", "Multilingual", "--ROUTING", routing,
                   "--NUM_LAYERS", "1", "--TEACHER_EN_CKPT", en, "--TEACHER_FR_CKPT", fr, "--BALANCE_LANG", "1")
        assert "avg teacher weights" in out
        r = _results(env, f"MTKD_{routing}_Multilingual_ENFR_S1F1_{tag}-L1_seed42")
        assert r["params_total"] < _results(env, f"FT_Multilingual_ENFR_S1F1_{tag}_seed42")["params_total"]
        assert r["efficiency"]["avg_epoch_train_sec"] is not None

    # a teacher from another fold has seen this run's test speakers → refused before data loading
    cmd = [sys.executable, os.path.join(PROJECT, "main.py"), "--BACKBONE", env["backbone"], "--SAVE_DIR", env["save"],
           "--PARADIGM", "MTKD", "--LINGUALITY", "Multilingual", "--CAFE_FOLD", "2",
           "--TEACHER_EN_CKPT", en, "--TEACHER_FR_CKPT", fr]
    r = subprocess.run(cmd, env=env["env"], cwd=PROJECT, capture_output=True, text=True)
    assert r.returncode != 0 and "was trained with CAFE_FOLD=1" in r.stderr, r.stderr[-3000:]

    # second run reuses the teacher cache
    out = _run(env, "--PARADIGM", "MTKD", "--LINGUALITY", "Multilingual", "--ROUTING", "language",
               "--NUM_LAYERS", "1", "--TEACHER_EN_CKPT", en, "--TEACHER_FR_CKPT", fr, "--SEED", "7")
    assert "using cached logits" in out

    # evaluation only (architecture rebuilt from the checkpoint's model_config)
    student = os.path.join(env["save"], f"MTKD_language_Multilingual_ENFR_S1F1_{tag}-L1_seed42.pth")
    out = _run(env, "--PARADIGM", "MTKD", "--LINGUALITY", "Multilingual", "--TRAINING", "0",
               "--STUDENT_CKPT", student)
    assert "=== TEST" in out

    # fixed time budget (autoresearch): stops on the clock, not N_EPOCHS
    out = _run(env, "--PARADIGM", "FT", "--LANGUAGE", "FR", "--RUN_NAME", "budget_run",
               "--TIME_BUDGET_MIN", "0.0005", "--N_EPOCHS", "50", "--PATIENCE", "50")
    assert "Time budget of" in out
    assert _results(env, "budget_run")["efficiency"]["epochs_run"] < 50

    # resume of a finished run → no training, just test
    _run(env, "--PARADIGM", "FT", "--LANGUAGE", "EN", "--RESUME", "1")

    r = subprocess.run([sys.executable, os.path.join(PROJECT, "summarize_results.py"), "--SAVE_DIR", env["save"],
                        "--group"], capture_output=True, text=True, cwd=PROJECT)
    assert r.returncode == 0, r.stderr
    assert "MTKD" in r.stdout
