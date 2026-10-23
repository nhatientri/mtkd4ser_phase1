"""
checkpoints.py – the checkpoint file format, loading, and resolution of local paths / W&B artifacts.

A checkpoint is a dict built by checkpoint_payload(); models.load_model_from_checkpoint and
pipeline.Trainer read it back through load_checkpoint(). W&B run logging lives in tracking.py.
"""

import os
import shutil

import torch


def ensure_dir(path):
    """Create `path` (a directory) or the parent directory of `path` (a file)."""
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path,
                exist_ok=True)


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------

def checkpoint_payload(model, optimizer, scheduler, scaler, *, epoch, select_key, val, best,
                       no_improve, is_best, config: dict) -> dict:
    """Everything needed to rebuild the model (model_config) and to resume training."""
    return {
        "epoch": epoch,
        "model_state_dict": model.model.state_dict(),
        "model_config": model.model_config,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "select_key": select_key,
        "val_select": val[select_key],
        "val_ur": val["UR"],
        "best_select": best,
        "best_ur": best,  # kept for older tooling
        "epochs_no_improve": no_improve,
        "is_best": is_best,
        "config": config,
    }


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=device)


def checkpoint_config(path) -> dict | None:
    """The training config stored in a checkpoint. Memory-mapped, so the weights are not read."""
    try:
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):  # torch < 2.1, or a legacy (non-zip) checkpoint file
        ckpt = load_checkpoint(path, "cpu")
    return ckpt.get("config")


# ---------------------------------------------------------------------------
# Resolution: local path or W&B artifact → local .pth
# ---------------------------------------------------------------------------

def _candidates(clean_ref: str) -> list[str]:
    """Artifact names to try for a reference (a .pth path is mapped to its run's artifact)."""
    is_path = ("/" in clean_ref or "\\" in clean_ref) and ":" not in os.path.basename(clean_ref)
    if clean_ref.endswith(".pth") or is_path:  # 'project/name:alias' is an artifact, not a path
        base_name = os.path.splitext(os.path.basename(clean_ref))[0]
        return [
            f"{base_name}-best:best",
            f"{base_name}-best:latest",
            f"{base_name}:best",
            f"{base_name}:latest",
            f"model-{base_name}:best",
            f"model-{base_name}:latest",
        ]
    if ":" in clean_ref:
        return [clean_ref]
    return [f"{clean_ref}:best", f"{clean_ref}:latest", clean_ref]


def _artifact_refs(cand: str, entity: str | None, project: str) -> list[str]:
    """Fully qualified variants of `cand` ('name', 'project/name' or 'entity/project/name')."""
    parts = cand.split("/")
    refs = []
    if len(parts) == 1:
        if entity:
            refs.append(f"{entity}/{project}/{cand}")
        refs.append(f"{project}/{cand}")
    elif len(parts) == 2 and entity:
        refs.append(f"{entity}/{cand}")
    refs.append(cand)
    return refs


def _link_local(wandb_run, candidates, entity, project):
    """Register an already-local checkpoint as an input artifact of `wandb_run` (lineage only)."""
    for cand in candidates:
        for r in _artifact_refs(cand, entity, project):
            try:
                wandb_run.use_artifact(r, type="model")
                print(f"  ✓ Linked W&B input artifact: {r}")
                return
            except Exception:
                continue


def _find_pth(download_dir):
    if os.path.isdir(download_dir):
        for root_dir, _, files in os.walk(download_dir):
            for fname in sorted(files):
                if fname.endswith(".pth"):
                    return os.path.join(root_dir, fname)
    elif os.path.isfile(download_dir) and download_dir.endswith(".pth"):
        return download_dir
    return None


def resolve_checkpoint(path_or_artifact, save_dir="/content/checkpoints",
                       wandb_run=None, default_project="mtkd4ser-phase1",
                       default_entity=None):
    """
    Resolves a checkpoint path. If path_or_artifact is:
    1. An existing local file: returns it directly (and links to wandb_run if active).
    2. A W&B Artifact URI (e.g. 'entity/project/name:alias', 'name:alias', or 'name'):
       downloads it via W&B, links it to wandb_run (for lineage tracking), and returns local .pth file path.
    3. A non-existent local file path (e.g. '/content/checkpoints/FT_Monolingual_EN_S1.pth'):
       derives candidate artifact names (e.g. 'FT_Monolingual_EN_S1-best:best') and automatically
       fetches the remote artifact from W&B.
    """
    if not path_or_artifact:
        return None

    # If it's already an existing local file and no wandb_run lineage registration is requested:
    if os.path.isfile(path_or_artifact) and wandb_run is None:
        return path_or_artifact

    clean_ref = str(path_or_artifact).replace("wandb://", "").strip()

    # Determine default entity from parameter, run, or environment
    if not default_entity:
        if wandb_run is not None and getattr(wandb_run, "entity", None):
            default_entity = wandb_run.entity
        else:
            default_entity = os.environ.get("WANDB_ENTITY") or None

    candidates = _candidates(clean_ref)

    # If file exists locally and we only need to link to active wandb_run:
    if os.path.isfile(path_or_artifact):
        _link_local(wandb_run, candidates, default_entity, default_project)
        return path_or_artifact

    # Otherwise, download from W&B
    try:
        import wandb
    except ImportError:
        print(f"[W&B Warning] 'wandb' package not installed; cannot resolve remote checkpoint: {clean_ref}")
        return path_or_artifact

    api = None
    if wandb_run is None:
        try:
            api = wandb.Api()
            if not default_entity and hasattr(api, "default_entity") and api.default_entity:
                default_entity = api.default_entity
        except Exception:
            pass

    artifact = None
    resolved_ref = None
    for cand in candidates:
        for r in _artifact_refs(cand, default_entity, default_project):
            try:
                if wandb_run is not None:
                    artifact = wandb_run.use_artifact(r, type="model")
                    print(f"  ✓ Linked W&B input artifact: {r}")
                elif api is not None:
                    artifact = api.artifact(r, type="model")
                if artifact is not None:
                    resolved_ref = r
                    break
            except Exception:
                continue
        if artifact is not None:
            break

    if artifact is not None:
        try:
            ensure_dir(save_dir)
            safe_artifact_name = artifact.name.replace(":", "_").replace("/", "_")
            artifact_dir = os.path.join(save_dir, "artifacts", safe_artifact_name)
            ensure_dir(artifact_dir)
            pth_file = _find_pth(artifact.download(root=artifact_dir))

            if pth_file and os.path.isfile(pth_file):
                dest_file = os.path.join(save_dir, os.path.basename(pth_file))
                if os.path.abspath(pth_file) != os.path.abspath(dest_file):
                    try:
                        shutil.copy2(pth_file, dest_file)
                        pth_file = dest_file
                    except Exception:
                        pass
                if clean_ref.endswith(".pth"):
                    req_path = clean_ref if os.path.isabs(clean_ref) else os.path.join(save_dir, os.path.basename(clean_ref))
                    if os.path.abspath(pth_file) != os.path.abspath(req_path):
                        try:
                            shutil.copy2(pth_file, req_path)
                        except Exception:
                            pass
                print(f"  ✓ Downloaded W&B checkpoint: {pth_file} (from {resolved_ref})")
                return pth_file
        except Exception as e:
            print(f"[W&B Warning] Failed to download artifact '{clean_ref}': {e}")

    return path_or_artifact
