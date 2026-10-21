"""
tracking.py – Weights & Biases run logging (init, teacher lineage, model artifacts, test metrics).

Every helper is a no-op-on-failure: W&B problems print a warning and training continues
with TensorBoard only.
"""

from typing import Any

from .checkpoints import resolve_checkpoint
from .config import Config

TEACHER_FIELDS = ("TEACHER_CKPT", "TEACHER_EN_CKPT", "TEACHER_FR_CKPT")


def init_wandb(cfg: Config, default_name: str):
    """Start a W&B run if cfg.USE_WANDB; returns the run or None."""
    if not cfg.USE_WANDB:
        return None
    try:
        import wandb
        run_id = cfg.WANDB_RUN_ID or None
        run = wandb.init(
            project=cfg.WANDB_PROJECT,
            entity=cfg.WANDB_ENTITY or None,
            name=cfg.WANDB_NAME or default_name,
            id=run_id,
            resume="allow" if run_id else None,
            config=cfg.to_dict(),
            reinit=True,
        )
        print(f"✓ Initialized Weights & Biases run: {run.name} ({run.url})")
        return run
    except ImportError:
        print("[W&B Warning] 'wandb' package not installed. Continuing with TensorBoard only.")
    except Exception as e:
        print(f"[W&B Warning] Failed to initialize W&B ({e}). Continuing with TensorBoard only.")
    return None


def track_teacher_artifacts(run, cfg: Config):
    """Register teacher artifacts in W&B so they appear in the project lineage graph."""
    if not run:
        return
    for field in TEACHER_FIELDS:
        ref = getattr(cfg, field)
        if ref:
            resolve_checkpoint(ref, save_dir=cfg.SAVE_DIR, wandb_run=run,
                               default_project=cfg.WANDB_PROJECT, default_entity=cfg.WANDB_ENTITY)


def log_model_artifact(run, ckpt_path, run_name, epoch, val_ur, best_ur=None, paradigm="Model", is_best=False):
    """Upload model checkpoint to W&B Artifacts with epoch-specific and status aliases."""
    if best_ur is None:
        best_ur = val_ur
    try:
        import wandb
        artifact_name = f"{run_name}-best".replace("/", "_").replace(" ", "_")
        aliases = [f"epoch_{epoch}", f"epoch-{epoch}", "latest"]
        if is_best:
            aliases.append("best")
            desc = f"Best {paradigm} checkpoint (Epoch {epoch}, val UR={val_ur:.4f})"
        else:
            desc = f"{paradigm} checkpoint (Epoch {epoch}, val UR={val_ur:.4f}, best UR={best_ur:.4f})"

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=desc,
            metadata={
                "epoch": epoch,
                "val_ur": float(val_ur),
                "best_ur": float(best_ur),
                "is_best": bool(is_best),
                "paradigm": paradigm,
            },
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact, aliases=aliases)
        best_tag = " [BEST]" if is_best else ""
        print(f"  ✓ Logged checkpoint to W&B Artifacts ({artifact_name}:{aliases[0]}{best_tag})")
    except Exception as e:
        print(f"[W&B Warning] Failed to log artifact to W&B: {e}")


def log_test(run, ur, wr, acc, loss, y_true, y_pred, label_names=None):
    """Log final test evaluation metrics and confusion matrix to W&B."""
    try:
        import wandb
        test_metrics: dict[str, Any] = {
            "test/UR": float(ur),
            "test/WR": float(wr),
            "test/acc": float(acc),
            "test/loss": float(loss),
        }
        if label_names is not None and len(y_true) > 0:
            try:
                cm = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=label_names,
                )
                test_metrics["test/confusion_matrix"] = cm
            except Exception as e:
                print(f"[W&B Warning] Could not plot confusion matrix: {e}")
        run.log(test_metrics)
        run.summary.update(test_metrics)
    except Exception as e:
        print(f"[W&B Warning] Could not log test metrics to W&B: {e}")
