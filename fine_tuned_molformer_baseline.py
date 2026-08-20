"""Fine-tuned MoLFormer baseline for the external BBB benchmark."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

import main as exp
from molformer_compat import enable_molformer_compat


hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = exp.MOLFORMER_MODEL
MAX_LENGTH = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS_MAX = 4
PATIENCE = 2
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_AMP = DEVICE.type == "cuda"


class EncodedSmilesDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], y: np.ndarray, idx: np.ndarray, with_labels: bool = True):
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.input_ids = encodings["input_ids"].index_select(0, self.idx)
        self.attention_mask = encodings["attention_mask"].index_select(0, self.idx)
        self.with_labels = with_labels
        self.y = torch.as_tensor(y[idx], dtype=torch.long) if with_labels else torch.zeros(len(idx), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        out = {
            "input_ids": self.input_ids[item],
            "attention_mask": self.attention_mask[item],
        }
        if self.with_labels:
            out["labels"] = self.y[item]
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_smiles(tokenizer, smiles: list[str]) -> dict[str, torch.Tensor]:
    return tokenizer(
        smiles,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def train_finetuned_molformer(
    train_encodings: dict[str, torch.Tensor],
    y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    train_sub, val_sub = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=seed + 1201,
        stratify=y[train_idx],
    )
    train_ds = EncodedSmilesDataset(train_encodings, y, train_sub)
    val_ds = EncodedSmilesDataset(train_encodings, y, val_sub)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
    )

    enable_molformer_compat()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)

    pos = float(np.sum(y[train_sub] == 1))
    neg = float(np.sum(y[train_sub] == 0))
    class_weight = torch.tensor(
        [1.0, neg / max(pos, 1.0)],
        dtype=torch.float32,
        device=DEVICE,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -math.inf
    stale = 0

    def predict_subset(loader: DataLoader) -> np.ndarray:
        probs = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels", None)
                inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(**inputs).logits
                logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.nan_to_num(np.concatenate(probs).astype(float), nan=0.5, posinf=1.0, neginf=0.0)

    for _epoch in range(EPOCHS_MAX):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(**inputs).logits
                loss = nn.functional.cross_entropy(logits, labels, weight=class_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        val_prob = predict_subset(val_loader)
        if len(np.unique(y[val_sub])) >= 2 and np.isfinite(val_prob).all():
            val_auc = float(roc_auc_score(y[val_sub], val_prob))
        else:
            val_auc = -math.inf
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_finetuned(
    model: nn.Module,
    encodings: dict[str, torch.Tensor],
    y: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    ds = EncodedSmilesDataset(encodings, y, idx, with_labels=True)
    loader = DataLoader(
        ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
    )
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(**inputs).logits
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.nan_to_num(np.concatenate(probs).astype(float), nan=0.5, posinf=1.0, neginf=0.0)


def run_seed(
    seed: int,
    bbb: exp.MolBundle,
    b3db: exp.MolBundle,
    bbb_encodings: dict[str, torch.Tensor],
    b3db_encodings: dict[str, torch.Tensor],
    mondrian: bool,
) -> dict[str, float | None]:
    split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
    ext_split = exp.scaffold_split(b3db, seed + 101)
    external_idx_raw = ext_split["test"]
    external_idx = exp.exact_nonoverlap_indices(bbb.smiles, b3db, external_idx_raw)
    if len(external_idx) == 0 or len(np.unique(b3db.y[external_idx])) < 2:
        external_idx = external_idx_raw

    model = train_finetuned_molformer(bbb_encodings, bbb.y, split["train"], seed)
    p_cal = predict_finetuned(model, bbb_encodings, bbb.y, split["calibration"])
    p_test = predict_finetuned(model, bbb_encodings, bbb.y, split["test"])
    p_external = predict_finetuned(model, b3db_encodings, b3db.y, external_idx)

    if mondrian:
        thresholds = exp.chemical_context_thresholds(bbb, split["train"], "basic")
        cal_ctx = exp.chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, "basic")
        test_ctx = exp.chemical_contexts_from_thresholds(bbb, split["test"], thresholds, "basic")
        ext_ctx = exp.chemical_contexts_from_thresholds(b3db, external_idx, thresholds, "basic")
        calibration = exp.calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=25)
    else:
        test_ctx = None
        ext_ctx = None
        calibration = exp.calibrate_global(bbb.y[split["calibration"]], p_cal)

    out = exp.evaluate_split(bbb.y[split["test"]], p_test, calibration, test_ctx)
    out.update(exp.evaluate_split(b3db.y[external_idx], p_external, calibration, ext_ctx, prefix="ood_external"))
    out["n_train"] = float(len(split["train"]))
    out["n_calibration"] = float(len(split["calibration"]))
    out["n_test"] = float(len(split["test"]))
    out["n_external_test"] = float(len(external_idx))
    out["n_external_exact_overlap_removed"] = float(len(external_idx_raw) - len(external_idx))
    out["mondrian_groups"] = float(len(calibration.get("groups", {})) if mondrian else 0)
    out["context_min_group"] = float(25 if mondrian else 0)
    out["context_scheme_basic"] = float(1 if mondrian else 0)
    out["molformer_finetune_epochs_max"] = float(EPOCHS_MAX)
    out["molformer_finetune_lr"] = float(LEARNING_RATE)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {k: exp.safe_number(v) for k, v in out.items()}


def main() -> None:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    enable_molformer_compat()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    bbb_encodings = encode_smiles(tokenizer, bbb.smiles)
    b3db_encodings = encode_smiles(tokenizer, b3db.smiles)

    output: dict[str, object] = {
        "status": "completed",
        "device": str(DEVICE),
        "model": MODEL_NAME,
        "method": "supervised fine-tuning of IBM MoLFormer-XL sequence classifier on BBB_Martins train folds",
        "hyperparameters": {
            "epochs_max": EPOCHS_MAX,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "amp": USE_AMP,
        },
        "conditions": {},
    }
    conditions = {
        "molformer_finetuned_global_cp": False,
        "molformer_finetuned_context_cp": True,
    }
    for condition, mondrian in conditions.items():
        rows = []
        for seed in exp.SEEDS:
            metrics = run_seed(seed, bbb, b3db, bbb_encodings, b3db_encodings, mondrian=mondrian)
            rows.append(metrics)
            exp.print_seed_metrics(condition, seed, metrics)
        summary = exp.summarize(rows)
        exp.print_summaries(condition, summary)
        output["conditions"][condition] = {
            "seeds_run": len(rows),
            "seed_metrics": rows,
            "summary": summary,
        }

    out_path = Path(__file__).resolve().parent / "fine_tuned_molformer_baseline.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
