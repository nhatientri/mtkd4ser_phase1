"""
pipeline.py – Phase 1
- Best-checkpoint selection on validation UR
- Early stopping
- Clean logging
- Colab-friendly save paths (/content/checkpoints/...)
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
# Fine-Tuning pipeline
# ---------------------------------------------------------------------------

def pipeline_ft(model, train_loader, valid_loader, test_loader,
                hyperparam, device):
    lr       = hyperparam["LEARNING_RATE"]
    n_epochs = hyperparam["N_EPOCHS"]
    training = hyperparam["TRAINING"]
    session  = hyperparam["SESSION"]
    ling     = hyperparam["LINGUALITY"]
    lang     = hyperparam["LANGUAGE"]
    save_dir = hyperparam.get("SAVE_DIR", "/content/checkpoints")
    patience = hyperparam.get("PATIENCE", 5)

    ckpt_path = os.path.join(save_dir, f"FT_{ling}_{lang}_S{session}.pth")
    _ensure_dir(ckpt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join(
        save_dir, "tb", f"FT_{ling}_{lang}_S{session}"))

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


# ---------------------------------------------------------------------------
# KD pipeline
# ---------------------------------------------------------------------------

def pipeline_kd(teacher, student, train_loader, valid_loader, test_loader,
                hyperparam, device):
    lr       = hyperparam["LEARNING_RATE"]
    n_epochs = hyperparam["N_EPOCHS"]
    training = hyperparam["TRAINING"]
    session  = hyperparam["SESSION"]
    ling     = hyperparam["LINGUALITY"]
    lang     = hyperparam["LANGUAGE"]
    save_dir = hyperparam.get("SAVE_DIR", "/content/checkpoints")
    patience = hyperparam.get("PATIENCE", 5)
    temperature = hyperparam.get("TEMPERATURE", 5.0)
    lambda_kd   = hyperparam.get("LAMBDA_KD", 0.25)

    ckpt_path = os.path.join(save_dir, f"KD_{ling}_{lang}_S{session}.pth")
    _ensure_dir(ckpt_path)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")

    writer = SummaryWriter(log_dir=os.path.join(
        save_dir, "tb", f"KD_{ling}_{lang}_S{session}"))

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
        ur, wr, acc, loss, *_ = eval_kd(
            teacher, student, tqdm(test_loader, desc="Test"),
            loss_ce, loss_kl, device, temperature, lambda_kd)
        print(f"Test UR={ur:.4f} WR={wr:.4f}")
        writer.close()
        return

    for epoch in range(start_epoch, n_epochs + 1):
        tr = train_kd(teacher, student,
                      tqdm(train_loader, desc=f"Ep {epoch} train"),
                      optimizer, loss_ce, loss_kl, device,
                      temperature, lambda_kd)
        va = eval_kd(teacher, student,
                     tqdm(valid_loader, desc=f"Ep {epoch} val"),
                     loss_ce, loss_kl, device, temperature, lambda_kd)

        tr_ur, tr_wr = tr[0], tr[1]
        va_ur, va_wr = va[0], va[1]
        print(f"Epoch {epoch}/{n_epochs}  train UR={tr_ur:.4f}  val UR={va_ur:.4f}")

        writer.add_scalar("UR/val", va_ur, epoch)
        writer.flush()

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
    print(f"\n=== FINAL TEST === UR={ur:.4f} WR={wr:.4f}")
    print_confusion(y_true, y_pred)
    writer.close()


# ---------------------------------------------------------------------------
# MTKD pipeline
# ---------------------------------------------------------------------------

def pipeline_mtkd(teachers, student, train_loader, valid_loader, test_loader,
                  hyperparam, device):
    lr       = hyperparam["LEARNING_RATE"]
    n_epochs = hyperparam["N_EPOCHS"]
    training = hyperparam["TRAINING"]
    session  = hyperparam["SESSION"]
    ling     = hyperparam["LINGUALITY"]
    lang     = hyperparam["LANGUAGE"]
    save_dir = hyperparam.get("SAVE_DIR", "/content/checkpoints")
    patience = hyperparam.get("PATIENCE", 5)
    temperature  = hyperparam.get("TEMPERATURE", 5.0)
    lambda_kd    = hyperparam.get("LAMBDA_KD", 0.25)
    routing_mode = hyperparam.get("ROUTING", "language")
    cosine_tau   = hyperparam.get("COSINE_TAU", 0.25)

    ckpt_path = os.path.join(
        save_dir, f"MTKD_{routing_mode}_{ling}_{lang}_S{session}.pth")
    _ensure_dir(ckpt_path)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")

    writer = SummaryWriter(log_dir=os.path.join(
        save_dir, "tb", f"MTKD_{routing_mode}_{ling}_{lang}_S{session}"))

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
        ur, wr, acc, loss, *_ = eval_mtkd(
            teachers, student, tqdm(test_loader, desc="Test"),
            loss_ce, loss_kl, device,
            routing_mode, temperature, lambda_kd, cosine_tau)
        print(f"Test UR={ur:.4f} WR={wr:.4f}")
        writer.close()
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

        tr_ur, va_ur = tr[0], va[0]
        print(f"Epoch {epoch}/{n_epochs}  train UR={tr_ur:.4f}  val UR={va_ur:.4f}")

        writer.add_scalar("UR/val", va_ur, epoch)
        writer.flush()

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
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model_state_dict"])
    ur, wr, acc, loss, _, _, y_true, y_pred = eval_mtkd(
        teachers, student, tqdm(test_loader, desc="Final Test"),
        loss_ce, loss_kl, device,
        routing_mode, temperature, lambda_kd, cosine_tau)
    print(f"\n=== FINAL TEST (routing={routing_mode}) ===")
    print(f"UR={ur:.4f}  WR={wr:.4f}  Acc={acc:.4f}")
    print_confusion(y_true, y_pred)
    writer.close()
