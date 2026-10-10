"""
utils.py – Phase 1 improvements
- language_id support in collate
- train_mtkd with 4 routing modes (language / uniform / cosine / confidence)
- teachers always eval() + no_grad
- single optimizer
- weights detached
- best-metric helpers
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import recall_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Label helpers (kept from original)
# ---------------------------------------------------------------------------

def update_iemocap_label(label):
    mapping = {"ang": "anger", "hap": "happiness", "exc": "happiness",
               "neu": "neutral", "sad": "sadness"}
    return mapping.get(label, label)


def update_fesc_label(label):
    mapping = {"1": "anger", "2": "happiness", "3": "neutral", "4": "sadness"}
    return mapping.get(str(label), label)


def is_common(emotion):
    return 1 if emotion in {"neutral", "sadness", "happiness", "anger"} else 0


def to_label(emo, label2id):
    return int(label2id[emo])


# ---------------------------------------------------------------------------
# Path helpers – Colab / local friendly
# Override these or pass absolute paths from data.py
# ---------------------------------------------------------------------------

def update_cafe_path(path, root="/content/CaFE"):
    """Make CaFE paths point to the extracted folder on Colab."""
    # original paths often contain long prefixes; keep only the filename part
    fname = path.split("/")[-1] if "/" in path else path
    return f"{root}/{fname}"


def update_iemocap_path(path, root="/content/IEMOCAP"):
    return path  # assume already correct after extraction


# ---------------------------------------------------------------------------
# Feature extraction & collate (now carries language_id)
# ---------------------------------------------------------------------------

def preprocess_function(examples, feature_extractor, max_duration=20.0):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        padding=True,
        truncation=True,
    )
    return inputs


def collate_fn(batch):
    """Collate that also returns language_id when present."""
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = pad_sequence(
        [torch.tensor(ex["input_values"]) for ex in batch],
        batch_first=True,
    )
    labels = torch.tensor([int(ex["label"]) for ex in batch])

    # language_id is optional for pure monolingual FT; default to 0
    if "language_id" in batch[0]:
        lang_ids = torch.tensor([int(ex["language_id"]) for ex in batch])
    else:
        lang_ids = torch.zeros(len(batch), dtype=torch.long)

    return {
        "input_values": inputs,
        "label": labels,
        "language_id": lang_ids,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_recalls(y_true, y_pred):
    ur = recall_score(y_true, y_pred, average="macro", zero_division=0)
    wr = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    return ur, wr


def print_confusion(y_true, y_pred, id2label=None):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    return cm


# ---------------------------------------------------------------------------
# Fine-tuning (FT) loops – unchanged logic, cleaner
# ---------------------------------------------------------------------------

def train_ft(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, all_true, all_pred = 0.0, [], []

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x).logits
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    ur, wr = compute_recalls(all_true, all_pred)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)
    return ur, wr, acc, total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_ft(model, loader, loss_fn, device):
    model.eval()
    total_loss, all_true, all_pred = 0.0, [], []

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)
        logits = model(x).logits
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    ur, wr = compute_recalls(all_true, all_pred)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)
    return ur, wr, acc, total_loss / max(len(loader), 1), all_true, all_pred


# ---------------------------------------------------------------------------
# Knowledge Distillation (single teacher)
# ---------------------------------------------------------------------------

def train_kd(teacher, student, loader, optimizer, loss_fn_ce, loss_fn_kl,
             device, temperature=5.0, lambda_kd=0.25):
    teacher.eval()
    student.train()
    total_loss = total_kl = total_ce = 0.0
    all_true, all_pred = [], []

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            t_logits = teacher(x).logits
            t_soft = F.softmax(t_logits / temperature, dim=-1)

        s_logits = student(x).logits
        s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)

        loss_kl = loss_fn_kl(s_log_soft, t_soft) * (temperature ** 2)
        loss_ce = loss_fn_ce(s_logits, y)
        loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl += loss_kl.item()
        total_ce += loss_ce.item()
        preds = s_logits.argmax(dim=-1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    n = max(len(loader), 1)
    ur, wr = compute_recalls(all_true, all_pred)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)
    return ur, wr, acc, total_loss / n, total_kl / n, total_ce / n


@torch.no_grad()
def eval_kd(teacher, student, loader, loss_fn_ce, loss_fn_kl,
            device, temperature=5.0, lambda_kd=0.25):
    teacher.eval()
    student.eval()
    total_loss = total_kl = total_ce = 0.0
    all_true, all_pred = [], []

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)

        t_logits = teacher(x).logits
        t_soft = F.softmax(t_logits / temperature, dim=-1)
        s_logits = student(x).logits
        s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)

        loss_kl = loss_fn_kl(s_log_soft, t_soft) * (temperature ** 2)
        loss_ce = loss_fn_ce(s_logits, y)
        loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kl

        total_loss += loss.item()
        total_kl += loss_kl.item()
        total_ce += loss_ce.item()
        preds = s_logits.argmax(dim=-1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    n = max(len(loader), 1)
    ur, wr = compute_recalls(all_true, all_pred)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)
    return ur, wr, acc, total_loss / n, total_kl / n, total_ce / n, all_true, all_pred


# ---------------------------------------------------------------------------
# Multi-Teacher KD – THE CORE OF PHASE 1
# ---------------------------------------------------------------------------

def _cosine_sim_batch(a, b):
    """Mean cosine similarity over the batch (detached)."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).mean().detach()


# Canonical language_id mapping: 0=EN, 1=FI, 2=FR
LANG_ID_TO_KEY = {0: "en", 1: "fi", 2: "fr"}


def _compute_routing_weights(routing_mode, lang_keys, t_logits, t_soft,
                             s_logits, lang_ids, device, cosine_tau=0.25):
    """
    Dynamically compute routing weights of shape (B, num_teachers) over active teachers.
    lang_keys: list of active language codes, e.g. ["en", "fr"]
    """
    B = s_logits.size(0)
    num_teachers = len(lang_keys)

    if routing_mode == "uniform":
        weights = torch.ones(B, num_teachers, device=device) / float(num_teachers)

    elif routing_mode == "cosine":
        cs = torch.stack([
            _cosine_sim_batch(s_logits, t_logits[k]).expand(B)
            for k in lang_keys
        ], dim=1)  # (B, num_teachers)
        weights = F.softmax(cs / cosine_tau, dim=-1)

    elif routing_mode == "confidence":
        conf = torch.stack([
            t_soft[k].max(dim=-1).values
            for k in lang_keys
        ], dim=1)  # (B, num_teachers)
        weights = F.softmax(conf, dim=-1)

    elif routing_mode == "language":
        # Oracle routing: map each sample's language_id to its teacher column
        weights = torch.zeros(B, num_teachers, device=device)
        lang2col = {k: i for i, k in enumerate(lang_keys)}

        max_lid = max(max(LANG_ID_TO_KEY.keys()), int(lang_ids.max().item() if lang_ids.numel() > 0 else 0))
        lid_to_col = torch.full((max_lid + 1,), -1, dtype=torch.long, device=device)
        for lid, key in LANG_ID_TO_KEY.items():
            if key in lang2col:
                lid_to_col[lid] = lang2col[key]

        col_indices = lid_to_col[lang_ids.clamp(0, max_lid)]
        matched = col_indices >= 0
        if matched.any():
            weights[matched, col_indices[matched]] = 1.0
        # Fallback to uniform if an unexpected language_id occurs
        if (~matched).any():
            weights[~matched] = 1.0 / float(num_teachers)

    else:
        raise ValueError(f"Unknown routing_mode: {routing_mode}")

    return weights.detach()


def train_mtkd(
    teachers,               # dict of active models, e.g. {"en": m, "fr": m}
    student,
    loader,
    optimizer,
    loss_fn_ce,
    loss_fn_kl,
    device,
    routing_mode="language",   # "language" | "uniform" | "cosine" | "confidence"
    temperature=5.0,
    lambda_kd=0.25,
    cosine_tau=0.25,
):
    """
    Language-aware (or ablation) multi-teacher knowledge distillation.
    Dynamically routes across only active teachers (e.g., EN+FR or EN+FI+FR).
    """
    for t in teachers.values():
        t.eval()
    student.train()

    lang_keys = list(teachers.keys())
    num_teachers = len(lang_keys)

    total_loss = total_kl = total_ce = 0.0
    all_true, all_pred = [], []
    weight_sum = torch.zeros(num_teachers, device=device)
    n_batches = 0

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)
        lang_ids = batch["language_id"].to(device)   # (B,)

        # ---- teacher forward (no grad) - only active teachers ----
        with torch.no_grad():
            t_logits = {k: teachers[k](x).logits for k in lang_keys}
            t_soft = {k: F.softmax(t_logits[k] / temperature, dim=-1) for k in lang_keys}

        # ---- student forward ----
        s_logits = student(x).logits
        s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)

        # ---- dynamic routing weights (B, num_teachers) ----
        weights = _compute_routing_weights(
            routing_mode, lang_keys, t_logits, t_soft, s_logits,
            lang_ids, device, cosine_tau
        )

        # ---- dynamic KL losses ----
        loss_kl = 0.0
        for i, k in enumerate(lang_keys):
            kl_k = loss_fn_kl(s_log_soft, t_soft[k]) * (temperature ** 2)
            loss_kl = loss_kl + weights[:, i].mean() * kl_k

        loss_ce = loss_fn_ce(s_logits, y)
        loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl += loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl
        total_ce += loss_ce.item()
        preds = s_logits.argmax(dim=-1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

        weight_sum += weights.mean(dim=0)
        n_batches += 1

    n = max(n_batches, 1)
    ur, wr = compute_recalls(all_true, all_pred)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)

    avg_w = (weight_sum / n).cpu().tolist()
    weight_str = "  ".join([f"{lang_keys[i].upper()}={avg_w[i]:.3f}" for i in range(num_teachers)])
    print(f"  [routing={routing_mode}] avg weights  {weight_str}")
    avg_w_dict = {f"weight_{lang_keys[i]}": avg_w[i] for i in range(num_teachers)}

    return ur, wr, acc, total_loss / n, total_kl / n, total_ce / n, avg_w_dict


@torch.no_grad()
def eval_mtkd(
    teachers,
    student,
    loader,
    loss_fn_ce,
    loss_fn_kl,
    device,
    routing_mode="language",
    temperature=5.0,
    lambda_kd=0.25,
    cosine_tau=0.25,
):
    for t in teachers.values():
        t.eval()
    student.eval()

    lang_keys = list(teachers.keys())
    num_teachers = len(lang_keys)

    total_loss = total_kl = total_ce = 0.0
    all_true, all_pred = [], []
    weight_sum = torch.zeros(num_teachers, device=device)
    n_batches = 0

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)
        lang_ids = batch["language_id"].to(device)

        t_logits = {k: teachers[k](x).logits for k in lang_keys}
        t_soft = {k: F.softmax(t_logits[k] / temperature, dim=-1) for k in lang_keys}

        s_logits = student(x).logits
        s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)

        weights = _compute_routing_weights(
            routing_mode, lang_keys, t_logits, t_soft, s_logits,
            lang_ids, device, cosine_tau
        )

        loss_kl = 0.0
        for i, k in enumerate(lang_keys):
            kl_k = loss_fn_kl(s_log_soft, t_soft[k]) * (temperature ** 2)
            loss_kl = loss_kl + weights[:, i].mean() * kl_k

        loss_ce = loss_fn_ce(s_logits, y)
        loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kl

        total_loss += loss.item()
        total_kl += loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl
        total_ce += loss_ce.item()
        preds = s_logits.argmax(dim=-1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        weight_sum += weights.mean(dim=0)
        n_batches += 1

    n = max(n_batches, 1)
    ur, wr = compute_recalls(all_true, all_pred)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)
    avg_w = (weight_sum / n).cpu().tolist()
    weight_str = "  ".join([f"{lang_keys[i].upper()}={avg_w[i]:.3f}" for i in range(num_teachers)])
    print(f"  [val routing={routing_mode}] avg weights  {weight_str}")
    avg_w_dict = {f"weight_{lang_keys[i]}": avg_w[i] for i in range(num_teachers)}

    return ur, wr, acc, total_loss / n, total_kl / n, total_ce / n, all_true, all_pred, avg_w_dict
