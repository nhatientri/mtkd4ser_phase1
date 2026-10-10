"""
pipeline.py – Phase 1
- Best-checkpoint selection on validation UR
- Early stopping
- Clean logging
- Colab-friendly save paths (/content/checkpoints/...)
- Weights & Biases (W&B) live metrics, confusion matrix, and artifact checkpointing
"""

import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import (
    train_ft, eval_ft,
    train_kd, eval_kd,
    train_mtkd, eval_mtkd,
    print_confusion,
)


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path,
                exist_ok=True)


# ---------------------------------------------------------------------------
# Weights & Biases (W&B) Helpers
# ---------------------------------------------------------------------------

def _init_wandb(hyperparam, default_name):
    """Safely initialize W&B if requested in hyperparam."""
    use_wandb = bool(hyperparam.get("USE_WANDB", 0))
    if not use_wandb:
        return False, None
    try:
        import wandb
        run_name = hyperparam.get("WANDB_NAME") or default_name
        # Keep config clean and serializable
        clean_config = {
            k: v for k, v in hyperparam.items()
            if not str(k).startswith("_") and not callable(v)
        }
        run = wandb.init(
            project=hyperparam.get("WANDB_PROJECT", "mtkd4ser-phase1"),
            entity=hyperparam.get("WANDB_ENTITY") or None,
            name=run_name,
            config=clean_config,
            reinit=True,
        )
        print(f"✓ Initialized Weights & Biases run: {run.name} ({run.url})")
        return True, run
    except ImportError:
        print("[W&B Warning] 'wandb' package not installed. Continuing with TensorBoard only.")
        return False, None
    except Exception as e:
        print(f"[W&B Warning] Failed to initialize W&B ({e}). Continuing with TensorBoard only.")
        return False, None


def _save_wandb_artifact(run, ckpt_path, run_name, epoch, best_ur, paradigm):
    """Upload the best model checkpoint to W&B Artifacts."""
    try:
        import wandb
        artifact_name = f"{run_name}-best".replace("/", "_").replace(" ", "_")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Best {paradigm} checkpoint (Epoch {epoch}, val UR={best_ur:.4f})",
            metadata={
                "epoch": epoch,
                "best_ur": float(best_ur),
                "paradigm": paradigm,
            },
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact, aliases=["best", "latest"])
        print(f"  ✓ Logged best checkpoint to W&B Artifacts ({artifact_name})")
    except Exception as e:
        print(f"[W&B Warning] Failed to log artifact to W&B: {e}")


def _log_test_wandb(run, ur, wr, acc, loss, y_true, y_pred, label_names=None):
    """Log final test evaluation metrics and confusion matrix to W&B."""
    try:
        import wandb
        test_metrics = {
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


# ---------------------------------------------------------------------------
# Fine-Tuning pipeline
# ---------------------------------------------------------------------------

def pipeline_ft(model, train_loader, valid_loader, test_loader,
                hyperparam, device):
    lr        = hyperparam["LEARNING_RATE"]
    n_epochs  = hyperparam["N_EPOCHS"]
    training  = hyperparam["TRAINING"]
    session   = hyperparam["SESSION"]
    ling      = hyperparam["LINGUALITY"]
    lang      = hyperparam["LANGUAGE"]
    save_dir  = hyperparam.get("SAVE_DIR", "/content/checkpoints")
    patience  = hyperparam.get("PATIENCE", 5)
    log_model = bool(hyperparam.get("WANDB_LOG_MODEL", 1))

    default_name = f"FT_{ling}_{lang}_S{session}"
    ckpt_path = os.path.join(save_dir, f"{default_name}.pth")
    _ensure_dir(ckpt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb", default_name))
    use_wandb, wandb_run = _init_wandb(hyperparam, default_name)

    start_epoch = 1
    best_ur = -1.0
    epochs_no_improve = 0

    if os.path.isfile(ckpt_path) and training:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_ur = ckpt.get("best_ur", -1.0)
        print(f"[FT] Resumed from epoch {start_epoch-1}, best_ur={best_ur:.4f}")

    if not training:
        ur, wr, acc, loss, y_true, y_pred = eval_ft(
            model, tqdm(test_loader, desc="Test"), loss_fn, device)
        print(f"Test  UR={ur:.4f}  WR={wr:.4f}  Acc={acc:.4f}  Loss={loss:.4f}")
        print_confusion(y_true, y_pred)
        writer.close()
        if use_wandb and wandb_run:
            _log_test_wandb(wandb_run, ur, wr, acc, loss, y_true, y_pred, hyperparam.get("label_names"))
            wandb_run.finish()
        return

    for epoch in range(start_epoch, n_epochs + 1):
        tr_ur, tr_wr, tr_acc, tr_loss = train_ft(
            model, tqdm(train_loader, desc=f"Ep {epoch} train"),
            optimizer, loss_fn, device)
        va_ur, va_wr, va_acc, va_loss, _, _ = eval_ft(
            model, tqdm(valid_loader, desc=f"Ep {epoch} val"),
            loss_fn, device)

        print(f"Epoch {epoch}/{n_epochs}  "
              f"train UR={tr_ur:.4f} WR={tr_wr:.4f}  "
              f"val UR={va_ur:.4f} WR={va_wr:.4f}")

        writer.add_scalar("UR/train", tr_ur, epoch)
        writer.add_scalar("UR/val",   va_ur, epoch)
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val",   va_loss, epoch)
        writer.flush()

        if use_wandb and wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/UR": tr_ur,
                "train/WR": tr_wr,
                "train/acc": tr_acc,
                "train/loss": tr_loss,
                "val/UR": va_ur,
                "val/WR": va_wr,
                "val/acc": va_acc,
                "val/loss": va_loss,
                "best_val_UR": max(best_ur, va_ur),
            }, step=epoch)

        if va_ur > best_ur:
            best_ur = va_ur
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ur": best_ur,
            }, ckpt_path)
            print(f"  → New best model saved (val UR={best_ur:.4f})")
            if use_wandb and wandb_run and log_model:
                _save_wandb_artifact(wandb_run, ckpt_path, wandb_run.name, epoch, best_ur, "FT")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test with best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ur, wr, acc, loss, y_true, y_pred = eval_ft(
        model, tqdm(test_loader, desc="Final Test"), loss_fn, device)
    print(f"\n=== FINAL TEST (best val UR={best_ur:.4f}) ===")
    print(f"UR={ur:.4f}  WR={wr:.4f}  Acc={acc:.4f}  Loss={loss:.4f}")
    print_confusion(y_true, y_pred)
    writer.close()

    if use_wandb and wandb_run:
        _log_test_wandb(wandb_run, ur, wr, acc, loss, y_true, y_pred, hyperparam.get("label_names"))
        wandb_run.finish()


# ---------------------------------------------------------------------------
# KD pipeline
# ---------------------------------------------------------------------------

def pipeline_kd(teacher, student, train_loader, valid_loader, test_loader,
                hyperparam, device):
    lr          = hyperparam["LEARNING_RATE"]
    n_epochs    = hyperparam["N_EPOCHS"]
    training    = hyperparam["TRAINING"]
    session     = hyperparam["SESSION"]
    ling        = hyperparam["LINGUALITY"]
    lang        = hyperparam["LANGUAGE"]
    save_dir    = hyperparam.get("SAVE_DIR", "/content/checkpoints")
    patience    = hyperparam.get("PATIENCE", 5)
    temperature = hyperparam.get("TEMPERATURE", 5.0)
    lambda_kd   = hyperparam.get("LAMBDA_KD", 0.25)
    log_model   = bool(hyperparam.get("WANDB_LOG_MODEL", 1))

    default_name = f"KD_{ling}_{lang}_S{session}"
    ckpt_path = os.path.join(save_dir, f"{default_name}.pth")
    _ensure_dir(ckpt_path)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb", default_name))
    use_wandb, wandb_run = _init_wandb(hyperparam, default_name)

    start_epoch = 1
    best_ur = -1.0
    epochs_no_improve = 0

    if os.path.isfile(ckpt_path) and training:
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_ur = ckpt.get("best_ur", -1.0)
        print(f"[KD] Resumed from epoch {start_epoch-1}")

    if not training:
        ur, wr, acc, loss, _, _, y_true, y_pred = eval_kd(
            teacher, student, tqdm(test_loader, desc="Test"),
            loss_ce, loss_kl, device, temperature, lambda_kd)
        print(f"Test UR={ur:.4f} WR={wr:.4f} Acc={acc:.4f} Loss={loss:.4f}")
        print_confusion(y_true, y_pred)
        writer.close()
        if use_wandb and wandb_run:
            _log_test_wandb(wandb_run, ur, wr, acc, loss, y_true, y_pred, hyperparam.get("label_names"))
            wandb_run.finish()
        return

    for epoch in range(start_epoch, n_epochs + 1):
        tr = train_kd(teacher, student,
                      tqdm(train_loader, desc=f"Ep {epoch} train"),
                      optimizer, loss_ce, loss_kl, device,
                      temperature, lambda_kd)
        va = eval_kd(teacher, student,
                     tqdm(valid_loader, desc=f"Ep {epoch} val"),
                     loss_ce, loss_kl, device, temperature, lambda_kd)

        tr_ur, tr_wr, tr_acc, tr_loss = tr[0], tr[1], tr[2], tr[3]
        va_ur, va_wr, va_acc, va_loss = va[0], va[1], va[2], va[3]
        print(f"Epoch {epoch}/{n_epochs}  train UR={tr_ur:.4f}  val UR={va_ur:.4f}")

        writer.add_scalar("UR/train", tr_ur, epoch)
        writer.add_scalar("UR/val",   va_ur, epoch)
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val",   va_loss, epoch)
        writer.flush()

        if use_wandb and wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/UR": tr_ur,
                "train/WR": tr_wr,
                "train/acc": tr_acc,
                "train/loss": tr_loss,
                "train/loss_kl": tr[4],
                "train/loss_ce": tr[5],
                "val/UR": va_ur,
                "val/WR": va_wr,
                "val/acc": va_acc,
                "val/loss": va_loss,
                "val/loss_kl": va[4],
                "val/loss_ce": va[5],
                "best_val_UR": max(best_ur, va_ur),
            }, step=epoch)

        if va_ur > best_ur:
            best_ur = va_ur
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ur": best_ur,
            }, ckpt_path)
            print(f"  → New best (val UR={best_ur:.4f})")
            if use_wandb and wandb_run and log_model:
                _save_wandb_artifact(wandb_run, ckpt_path, wandb_run.name, epoch, best_ur, "KD")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state_dict"])
    ur, wr, acc, loss, _, _, y_true, y_pred = eval_kd(
        teacher, student, tqdm(test_loader, desc="Final Test"),
        loss_ce, loss_kl, device, temperature, lambda_kd)
    print(f"\n=== FINAL TEST === UR={ur:.4f} WR={wr:.4f} Acc={acc:.4f} Loss={loss:.4f}")
    print_confusion(y_true, y_pred)
    writer.close()

    if use_wandb and wandb_run:
        _log_test_wandb(wandb_run, ur, wr, acc, loss, y_true, y_pred, hyperparam.get("label_names"))
        wandb_run.finish()


# ---------------------------------------------------------------------------
# MTKD pipeline
# ---------------------------------------------------------------------------

def pipeline_mtkd(teachers, student, train_loader, valid_loader, test_loader,
                  hyperparam, device):
    lr           = hyperparam["LEARNING_RATE"]
    n_epochs     = hyperparam["N_EPOCHS"]
    training     = hyperparam["TRAINING"]
    session      = hyperparam["SESSION"]
    ling         = hyperparam["LINGUALITY"]
    lang         = hyperparam["LANGUAGE"]
    save_dir     = hyperparam.get("SAVE_DIR", "/content/checkpoints")
    patience     = hyperparam.get("PATIENCE", 5)
    temperature  = hyperparam.get("TEMPERATURE", 5.0)
    lambda_kd    = hyperparam.get("LAMBDA_KD", 0.25)
    routing_mode = hyperparam.get("ROUTING", "language")
    cosine_tau   = hyperparam.get("COSINE_TAU", 0.25)
    log_model    = bool(hyperparam.get("WANDB_LOG_MODEL", 1))

    default_name = f"MTKD_{routing_mode}_{ling}_{lang}_S{session}"
    ckpt_path = os.path.join(save_dir, f"{default_name}.pth")
    _ensure_dir(ckpt_path)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb", default_name))
    use_wandb, wandb_run = _init_wandb(hyperparam, default_name)

    start_epoch = 1
    best_ur = -1.0
    epochs_no_improve = 0

    if os.path.isfile(ckpt_path) and training:
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_ur = ckpt.get("best_ur", -1.0)
        print(f"[MTKD] Resumed from epoch {start_epoch-1}")

    if not training:
        ur, wr, acc, loss, _, _, y_true, y_pred, *_ = eval_mtkd(
            teachers, student, tqdm(test_loader, desc="Test"),
            loss_ce, loss_kl, device,
            routing_mode, temperature, lambda_kd, cosine_tau)
        print(f"Test UR={ur:.4f} WR={wr:.4f} Acc={acc:.4f} Loss={loss:.4f}")
        print_confusion(y_true, y_pred)
        writer.close()
        if use_wandb and wandb_run:
            _log_test_wandb(wandb_run, ur, wr, acc, loss, y_true, y_pred, hyperparam.get("label_names"))
            wandb_run.finish()
        return

    for epoch in range(start_epoch, n_epochs + 1):
        tr = train_mtkd(
            teachers, student,
            tqdm(train_loader, desc=f"Ep {epoch} train"),
            optimizer, loss_ce, loss_kl, device,
            routing_mode, temperature, lambda_kd, cosine_tau)
        va = eval_mtkd(
            teachers, student,
            tqdm(valid_loader, desc=f"Ep {epoch} val"),
            loss_ce, loss_kl, device,
            routing_mode, temperature, lambda_kd, cosine_tau)

        tr_ur, tr_wr, tr_acc, tr_loss = tr[0], tr[1], tr[2], tr[3]
        va_ur, va_wr, va_acc, va_loss = va[0], va[1], va[2], va[3]
        print(f"Epoch {epoch}/{n_epochs}  train UR={tr_ur:.4f}  val UR={va_ur:.4f}")

        writer.add_scalar("UR/train", tr_ur, epoch)
        writer.add_scalar("UR/val",   va_ur, epoch)
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val",   va_loss, epoch)
        writer.flush()

        if use_wandb and wandb_run:
            epoch_data = {
                "epoch": epoch,
                "train/UR": tr_ur,
                "train/WR": tr_wr,
                "train/acc": tr_acc,
                "train/loss": tr_loss,
                "train/loss_kl": tr[4],
                "train/loss_ce": tr[5],
                "val/UR": va_ur,
                "val/WR": va_wr,
                "val/acc": va_acc,
                "val/loss": va_loss,
                "val/loss_kl": va[4],
                "val/loss_ce": va[5],
                "best_val_UR": max(best_ur, va_ur),
            }
            if len(tr) > 6 and isinstance(tr[6], dict):
                for k, w in tr[6].items():
                    epoch_data[f"routing_train/{k}"] = w
            if len(va) > 8 and isinstance(va[8], dict):
                for k, w in va[8].items():
                    epoch_data[f"routing_val/{k}"] = w
            wandb_run.log(epoch_data, step=epoch)

        if va_ur > best_ur:
            best_ur = va_ur
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ur": best_ur,
                "routing_mode": routing_mode,
            }, ckpt_path)
            print(f"  → New best (val UR={best_ur:.4f})")
            if use_wandb and wandb_run and log_model:
                _save_wandb_artifact(wandb_run, ckpt_path, wandb_run.name, epoch, best_ur, "MTKD")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state_dict"])
    ur, wr, acc, loss, _, _, y_true, y_pred, *_ = eval_mtkd(
        teachers, student, tqdm(test_loader, desc="Final Test"),
        loss_ce, loss_kl, device,
        routing_mode, temperature, lambda_kd, cosine_tau)
    print(f"\n=== FINAL TEST (routing={routing_mode}) ===")
    print(f"UR={ur:.4f}  WR={wr:.4f}  Acc={acc:.4f}  Loss={loss:.4f}")
    print_confusion(y_true, y_pred)
    writer.close()

    if use_wandb and wandb_run:
        _log_test_wandb(wandb_run, ur, wr, acc, loss, y_true, y_pred, hyperparam.get("label_names"))
        wandb_run.finish()
