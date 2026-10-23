"""Checkpoint format and path / W&B-artifact resolution (offline – no W&B calls)."""

import torch

from mtkd4ser.checkpoints import (
    _artifact_refs, _candidates, checkpoint_config, checkpoint_payload, load_checkpoint, resolve_checkpoint,
)


def test_artifact_refs_name_only():
    assert _artifact_refs("run-best:best", "me", "proj") == ["me/proj/run-best:best", "proj/run-best:best",
                                                             "run-best:best"]
    assert _artifact_refs("run-best:best", None, "proj") == ["proj/run-best:best", "run-best:best"]


def test_artifact_refs_project_and_full():
    assert _artifact_refs("proj/x:best", "me", "p") == ["me/proj/x:best", "proj/x:best"]
    assert _artifact_refs("proj/x:best", None, "p") == ["proj/x:best"]
    assert _artifact_refs("me/proj/x:best", "other", "p") == ["me/proj/x:best"]


def test_candidates():
    pth = ["r-best:best", "r-best:latest", "r:best", "r:latest", "model-r:best", "model-r:latest"]
    assert _candidates("r.pth") == pth
    assert _candidates("/content/checkpoints/r.pth") == pth
    assert _candidates("name:v3") == ["name:v3"]
    assert _candidates("name") == ["name:best", "name:latest", "name"]
    # project- / entity-qualified artifact refs are passed through, not treated as file paths
    assert _candidates("proj/r-best:best") == ["proj/r-best:best"]
    assert _candidates("me/proj/r-best:v2") == ["me/proj/r-best:v2"]


def test_checkpoint_config(tmp_path):
    path = tmp_path / "c.pth"
    torch.save({"model_state_dict": {}, "config": {"SESSION": 2}}, path)
    assert checkpoint_config(str(path)) == {"SESSION": 2}
    torch.save({"model_state_dict": {}}, path)
    assert checkpoint_config(str(path)) is None


def test_resolve_existing_local_file(tmp_path):
    p = tmp_path / "x.pth"
    p.write_bytes(b"")
    assert resolve_checkpoint(str(p), save_dir=str(tmp_path)) == str(p)
    assert resolve_checkpoint(None) is None


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 4)
        self.model_config = {"backbone": "tiny", "num_layers": None}


def test_payload_roundtrip(tmp_path):
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    payload = checkpoint_payload(model, opt, sched, scaler, epoch=3, select_key="UR_lang_mean",
                                 val={"UR": 0.5, "UR_lang_mean": 0.6}, best=0.6, no_improve=0,
                                 is_best=True, config={"SEED": 42})
    path = tmp_path / "c.pth"
    torch.save(payload, path)
    ckpt = load_checkpoint(str(path), "cpu")
    # keys read by models.load_model_from_checkpoint and pipeline.Trainer (_try_resume / fit)
    for key in ("model_state_dict", "model_config", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "epoch", "best_select", "best_ur", "epochs_no_improve"):
        assert key in ckpt
    assert ckpt["epoch"] == 3 and ckpt["val_select"] == 0.6 and ckpt["val_ur"] == 0.5
    model2 = _Tiny()
    model2.model.load_state_dict(ckpt["model_state_dict"])
    assert torch.equal(model2.model.weight, model.model.weight)
