"""
pipeline.py – one training loop for FT, KD and MTKD.

  FT    : loss = CE
  KD    : loss = (1-λ)·CE + λ·KD(student, teacher)                      (K = 1 teacher)
  MTKD  : loss = (1-λ)·CE + λ·Σ_k w_k(sample)·KD(student, teacher_k)   (router decides w)

Teachers are frozen, in eval mode and see un-augmented audio, so their logits are the same
every epoch. They are therefore computed ONCE (compute_teacher_logits), cached to disk, and
the teachers are removed from GPU memory before the student starts training. Distillation
then costs about the same per epoch as plain fine-tuning, whatever the routing mode.
"""

import contextlib
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .checkpoints import checkpoint_payload, ensure_dir, load_checkpoint, resolve_checkpoint
from .config import Config, run_name
from .constants import LABELS, LANG_ID_TO_KEY, NUM_CLASSES
from .distill import Router, distill_loss
from .models import SERModel, load_teacher
from .tracking import init_wandb, log_model_artifact, log_test, track_teacher_artifacts
from .utils import collate_fn, compute_metrics, print_confusion

LOADER_COLUMNS = ["input_values", "label", "language_id", "idx"]


def _as_torch_dataset(ds):
    return ds.with_format("numpy", columns=[c for c in LOADER_COLUMNS if c in ds.column_names])


def _autocast(device, enabled):
    """fp16 autocast on CUDA only (MPS/CPU autocast support differs across torch versions)."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Teacher logit cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_logits(teacher_ckpt: str, dataset, device, cfg: Config) -> tuple[torch.Tensor, float]:
    """
    Returns (logits[N, C] indexed by the dataset's `idx` column, seconds spent).
    Cached under SAVE_DIR/teacher_cache/, keyed by checkpoint file + dataset fingerprint.
    """
    st = os.stat(teacher_ckpt)
    key_src = "|".join([os.path.abspath(teacher_ckpt), str(st.st_size), str(int(st.st_mtime)),
                        str(getattr(dataset, "_fingerprint", "")), str(len(dataset))])
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    cache_dir = os.path.join(cfg.SAVE_DIR, "teacher_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    if os.path.isfile(cache_path):
        print(f"[teacher] using cached logits {cache_path}")
        return torch.load(cache_path), 0.0

    t0 = time.time()
    teacher = load_teacher(teacher_ckpt, device, cfg.TEACHER_BACKBONE)
    loader = DataLoader(_as_torch_dataset(dataset), batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)
    logits = torch.zeros(len(dataset), NUM_CLASSES)
    labels = torch.zeros(len(dataset), dtype=torch.long)
    for batch in tqdm(loader, desc=f"teacher logits ({os.path.basename(teacher_ckpt)})"):
        x = batch["input_values"].to(device)
        m = batch["attention_mask"].to(device)
        with _autocast(device, cfg.AMP):
            out = teacher(x, m)
        logits[batch["idx"]] = out.float().cpu()
        labels[batch["idx"]] = batch["label"]
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _sync(device)
    secs = time.time() - t0
    torch.save(logits, cache_path)

    # Diagnostic: teachers usually saw these exact clips during their own training, so they
    # tend to be near-certain here – which limits how much "dark knowledge" they transfer.
    probs = logits.softmax(-1)
    acc = (probs.argmax(-1) == labels).float().mean().item()
    print(f"[teacher] {os.path.basename(teacher_ckpt)}: train-set acc={acc:.3f}, "
          f"mean max-prob={probs.max(-1).values.mean().item():.3f}  ({secs:.0f}s, cached to {cache_path})")
    return logits, secs


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Config, model: SERModel, splits: dict, device,
                 teacher_logits: torch.Tensor | None = None, router: Router | None = None,
                 teacher_cache_sec: float = 0.0):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.name = run_name(cfg)
        self.amp = bool(cfg.AMP and device.type == "cuda")
        self.teacher_logits = teacher_logits.to(device) if teacher_logits is not None else None  # (N,K,C)
        self.router = router
        self.teacher_cache_sec = teacher_cache_sec

        g = torch.Generator()
        g.manual_seed(cfg.SEED)
        train_ds = splits["train"]
        sampler = None
        if cfg.BALANCE_LANG:
            lids = np.asarray(list(train_ds["language_id"]))
            counts = {l: (lids == l).sum() for l in np.unique(lids)}
            weights = torch.tensor([1.0 / counts[l] for l in lids], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(lids), replacement=True, generator=g)
        pin = device.type == "cuda"
        self.train_loader = DataLoader(_as_torch_dataset(train_ds), batch_size=cfg.BATCH_SIZE,
                                       shuffle=sampler is None, sampler=sampler, generator=g,
                                       collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS, pin_memory=pin)
        self.dev_loader = DataLoader(_as_torch_dataset(splits["dev"]), batch_size=cfg.BATCH_SIZE,
                                     shuffle=False, collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)
        self.test_loader = DataLoader(_as_torch_dataset(splits["test"]), batch_size=cfg.BATCH_SIZE,
                                      shuffle=False, collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)

        dev_langs = {int(v) for v in splits["dev"]["language_id"]}
        if cfg.SELECT_METRIC == "auto":
            self.select_key = "UR_lang_mean" if len(dev_langs) > 1 else "UR"
        else:
            self.select_key = cfg.SELECT_METRIC

        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(self.train_loader) / max(cfg.ACCUM_STEPS, 1))
        total = max(steps_per_epoch * cfg.N_EPOCHS, 1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(cfg.WARMUP_RATIO * total), total)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):  # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.ce = torch.nn.CrossEntropyLoss()

        ensure_dir(cfg.SAVE_DIR)
        self.best_path = os.path.join(cfg.SAVE_DIR, f"{self.name}.pth")
        self.last_path = os.path.join(cfg.SAVE_DIR, f"{self.name}_last.pth")
        self.results_path = os.path.join(cfg.SAVE_DIR, "results", f"{self.name}.json")

        self.writer = None
        self.wandb_run = None

    # ------------------------------------------------------------------ steps
    def _batch_to(self, batch):
        d = self.device
        nb = d.type == "cuda"
        return (batch["input_values"].to(d, non_blocking=nb), batch["attention_mask"].to(d, non_blocking=nb),
                batch["label"].to(d, non_blocking=nb), batch["language_id"].to(d, non_blocking=nb))

    def train_epoch(self, epoch: int) -> dict:
        cfg, model = self.cfg, self.model
        model.train()
        accum = max(cfg.ACCUM_STEPS, 1)
        tot = {"loss": 0.0, "loss_ce": 0.0, "loss_kd": 0.0}
        y_true, y_pred, langs = [], [], []
        K = self.teacher_logits.shape[1] if self.teacher_logits is not None else 0
        w_sum = torch.zeros(K, device=self.device)
        n_seen = 0
        self.optimizer.zero_grad(set_to_none=True)

        n_batches = len(self.train_loader)
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Ep {epoch} train")):
            x, m, y, lid = self._batch_to(batch)
            with _autocast(self.device, cfg.AMP):
                s_logits = model(x, m)
            s_logits = s_logits.float()
            loss_ce = self.ce(s_logits, y)
            if self.teacher_logits is not None:
                t_logits = self.teacher_logits[batch["idx"].to(self.device)]          # (B,K,C)
                w = self.router(s_logits.detach(), t_logits, lid)                     # (B,K)
                loss_kd = distill_loss(s_logits, t_logits, w, cfg.TEMPERATURE)
                loss = (1.0 - cfg.LAMBDA_KD) * loss_ce + cfg.LAMBDA_KD * loss_kd
                w_sum += w.sum(0)
                tot["loss_kd"] += loss_kd.item()
            else:
                loss = loss_ce

            self.scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == n_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            tot["loss"] += loss.item()
            tot["loss_ce"] += loss_ce.item()
            y_true += y.tolist()
            y_pred += s_logits.argmax(-1).tolist()
            langs += lid.tolist()
            n_seen += len(y)

        n = max(n_batches, 1)
        out = compute_metrics(y_true, y_pred, langs)
        out.update({k: v / n for k, v in tot.items()})
        out["n_samples"] = n_seen
        if K and self.router is not None:
            avg_w = (w_sum / max(n_seen, 1)).tolist()
            for i, lang in enumerate(self.router.teacher_langs):
                out[f"weight_{lang}"] = avg_w[i]
            print("  [routing=%s] avg teacher weights  %s" % (
                self.router.name, "  ".join(f"{l.upper()}={w:.3f}" for l, w in zip(self.router.teacher_langs, avg_w))))
        return out

    @torch.no_grad()
    def evaluate(self, loader, desc="eval") -> dict:
        """Student only – teachers are never run at evaluation time."""
        self.model.eval()
        y_true, y_pred, langs, loss_sum = [], [], [], 0.0
        for batch in tqdm(loader, desc=desc):
            x, m, y, lid = self._batch_to(batch)
            with _autocast(self.device, self.cfg.AMP):
                logits = self.model(x, m)
            logits = logits.float()
            loss_sum += self.ce(logits, y).item()
            y_true += y.tolist()
            y_pred += logits.argmax(-1).tolist()
            langs += lid.tolist()
        out = compute_metrics(y_true, y_pred, langs)
        out["loss"] = loss_sum / max(len(loader), 1)
        out["_y_true"], out["_y_pred"], out["_langs"] = y_true, y_pred, langs
        return out

    # ------------------------------------------------------------------ checkpoints
    def _payload(self, epoch, val, best, no_improve, is_best):
        return checkpoint_payload(self.model, self.optimizer, self.scheduler, self.scaler,
                                  epoch=epoch, select_key=self.select_key, val=val, best=best,
                                  no_improve=no_improve, is_best=is_best, config=self.cfg.to_dict())

    def _try_resume(self):
        cfg = self.cfg
        target = cfg.RESUME_CKPT or (self.last_path if cfg.RESUME and os.path.isfile(self.last_path) else None)
        if not target:
            if os.path.isfile(self.last_path) and not cfg.RESUME:
                print(f"[info] {self.last_path} exists but RESUME=0 – starting a fresh run (it will be overwritten).")
            return 1, -1.0, 0
        path = resolve_checkpoint(target, save_dir=cfg.SAVE_DIR, wandb_run=self.wandb_run,
                                  default_project=cfg.WANDB_PROJECT, default_entity=cfg.WANDB_ENTITY)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Resume checkpoint not found: {target}")
        ckpt = load_checkpoint(path, self.device)
        self.model.model.load_state_dict(ckpt["model_state_dict"])
        for key, obj in (("optimizer_state_dict", self.optimizer), ("scheduler_state_dict", self.scheduler),
                         ("scaler_state_dict", self.scaler)):
            if key in ckpt:
                try:
                    obj.load_state_dict(ckpt[key])
                except Exception as e:
                    print(f"[warn] could not restore {key}: {e}")
        start = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best_select", ckpt.get("best_ur", -1.0)))
        no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"[resume] {path}: epoch {start - 1}, best {self.select_key}={best:.4f}")
        return start, best, no_improve

    # ------------------------------------------------------------------ logging
    def _log(self, data: dict, step: int):
        if self.writer is not None:
            for k, v in data.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    self.writer.add_scalar(k, v, step)
            self.writer.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(data, step=step)

    @staticmethod
    def _flat(prefix, metrics):
        return {f"{prefix}/{k}": v for k, v in metrics.items() if not k.startswith("_")}

    def _open_logs(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.SAVE_DIR, "tb", self.name))
        self.wandb_run = init_wandb(self.cfg, self.name)
        if self.wandb_run is not None:
            track_teacher_artifacts(self.wandb_run, self.cfg)
            self.wandb_run.summary.update({"params/total": self.model.num_parameters(),
                                           "params/trainable": self.model.num_parameters(True)})

    def _close_logs(self):
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    # ------------------------------------------------------------------ main loops
    def fit(self):
        cfg = self.cfg
        self._open_logs()
        start_epoch, best, no_improve = self._try_resume()
        best_epoch = None
        epoch_secs, epoch_samples, peak_mb = [], [], 0.0
        print(f"[train] {self.name} | params total={self.model.num_parameters()/1e6:.1f}M "
              f"trainable={self.model.num_parameters(True)/1e6:.1f}M | select on dev {self.select_key}")

        for epoch in range(start_epoch, cfg.N_EPOCHS + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _sync(self.device)
            t0 = time.time()
            tr = self.train_epoch(epoch)
            _sync(self.device)
            secs = time.time() - t0
            epoch_secs.append(secs)
            epoch_samples.append(tr["n_samples"])
            if self.device.type == "cuda":
                peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 2**20)

            va = self.evaluate(self.dev_loader, f"Ep {epoch} dev")
            sel = va[self.select_key]
            is_best = sel > best
            if is_best:
                best, no_improve, best_epoch = sel, 0, epoch
            else:
                no_improve += 1

            lang_str = "  ".join(f"{k}={v:.4f}" for k, v in va.items() if k.startswith("UR_") and k != "UR_lang_mean")
            print(f"Epoch {epoch}/{cfg.N_EPOCHS}  train UR={tr['UR']:.4f} loss={tr['loss']:.4f} | "
                  f"dev UR={va['UR']:.4f} WR={va['WR']:.4f} {lang_str}  "
                  f"[{self.select_key}={sel:.4f}{' *best*' if is_best else ''}]  ({secs:.0f}s)")

            log = {"epoch": epoch, **self._flat("train", tr), **self._flat("val", va),
                   "best_val_select": best, "time/epoch_sec": secs,
                   "time/train_samples_per_sec": tr["n_samples"] / max(secs, 1e-9),
                   "lr": self.scheduler.get_last_lr()[0]}
            if peak_mb:
                log["mem/peak_mb"] = peak_mb
            self._log(log, epoch)

            payload = self._payload(epoch, va, best, no_improve, is_best)
            torch.save(payload, self.last_path)
            if is_best:
                torch.save(payload, self.best_path)
            epoch_path = None
            if cfg.SAVE_ALL_EPOCHS:
                epoch_path = os.path.join(cfg.SAVE_DIR, f"{self.name}_epoch{epoch}.pth")
                torch.save(payload, epoch_path)
            if self.wandb_run is not None and cfg.WANDB_LOG_MODEL and (is_best or epoch_path):
                log_model_artifact(self.wandb_run, self.best_path if is_best else epoch_path, self.name,
                                     epoch, sel, best, cfg.PARADIGM, is_best=is_best)

            if no_improve >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        efficiency = {
            "epochs_run": len(epoch_secs),
            "avg_epoch_train_sec": float(np.mean(epoch_secs)) if epoch_secs else None,
            "train_samples_per_sec": float(np.sum(epoch_samples) / max(np.sum(epoch_secs), 1e-9)) if epoch_secs else None,
            "total_train_sec": float(np.sum(epoch_secs)) if epoch_secs else 0.0,
            "teacher_cache_sec": self.teacher_cache_sec,
            "peak_gpu_mem_mb": peak_mb or None,
        }
        if os.path.isfile(self.best_path):
            ckpt = load_checkpoint(self.best_path, self.device)
            self.model.model.load_state_dict(ckpt["model_state_dict"])
            best_epoch = ckpt.get("epoch", best_epoch)
        else:
            print(f"[warn] no best checkpoint at {self.best_path}; testing the current weights.")
        self.test(extra={"best_dev_select": best, "best_epoch": best_epoch, "efficiency": efficiency})

    def test(self, extra: dict | None = None, ckpt_path: str | None = None):
        if ckpt_path:  # evaluation-only: never overwrite the training run's results file
            base = os.path.splitext(os.path.basename(ckpt_path))[0]
            self.results_path = os.path.join(self.cfg.SAVE_DIR, "results", f"{base}_eval.json")
        if self.writer is None and self.wandb_run is None:
            self._open_logs()
        _sync(self.device)
        t0 = time.time()
        te = self.evaluate(self.test_loader, "Test")
        _sync(self.device)
        test_secs = time.time() - t0

        print(f"\n=== TEST  {self.name} ===")
        print(f"UR={te['UR']:.4f}  WR={te['WR']:.4f}  acc={te['acc']:.4f}  loss={te['loss']:.4f}  (n={te['n']})")
        y_true, y_pred, langs = map(np.asarray, (te["_y_true"], te["_y_pred"], te["_langs"]))
        print_confusion(y_true, y_pred, LABELS, "All languages")
        lang_ids = sorted(set(langs.tolist()))
        if len(lang_ids) > 1:
            for lid in lang_ids:
                key = LANG_ID_TO_KEY.get(lid, str(lid))
                print(f"  {key.upper()}: UR={te[f'UR_{key}']:.4f}  WR={te[f'WR_{key}']:.4f}  n={te[f'n_{key}']}")
                sel = langs == lid
                print_confusion(y_true[sel], y_pred[sel], LABELS, f"{key.upper()} only")
            print(f"  mean per-language UR = {te['UR_lang_mean']:.4f}")

        test_metrics = {k: v for k, v in te.items() if not k.startswith("_")}
        results = {
            "run_name": os.path.splitext(os.path.basename(self.results_path))[0],
            "config": self.cfg.to_dict(),
            "checkpoint": ckpt_path or self.best_path,
            "model_config": self.model.model_config,
            "params_total": self.model.num_parameters(),
            "params_trainable": self.model.num_parameters(True),
            "test": test_metrics,
            "test_ms_per_clip": 1000.0 * test_secs / max(te["n"], 1),
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else self.device.type,
            **(extra or {}),
        }
        ensure_dir(self.results_path)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[results] {self.results_path}")

        if self.wandb_run is not None:
            log_test(self.wandb_run, te["UR"], te["WR"], te["acc"], te["loss"],
                            te["_y_true"], te["_y_pred"], LABELS)
            extra_log = {f"test/{k}": v for k, v in test_metrics.items() if k.startswith(("UR_", "WR_", "acc_"))}
            extra_log["test/ms_per_clip"] = results["test_ms_per_clip"]
            self.wandb_run.summary.update(extra_log)
            if extra and extra.get("efficiency"):
                self.wandb_run.summary.update({f"efficiency/{k}": v for k, v in extra["efficiency"].items()
                                               if v is not None})
        self._close_logs()
        return results
