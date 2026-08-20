"""Compact message-passing GNN baselines for the external BBB benchmark."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import main as exp


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GNN_EPOCHS = 70
GNN_BATCH_SIZE = 128
GNN_LR = 1.5e-3
GNN_WEIGHT_DECAY = 1e-4
GNN_CONFIGS = [
    {"hidden_dim": 128, "dropout": 0.15, "num_layers": 3},
    {"hidden_dim": 192, "dropout": 0.10, "num_layers": 4},
    {"hidden_dim": 256, "dropout": 0.15, "num_layers": 4},
]


ATOM_NUMBERS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
]


@dataclass
class GraphBundle:
    x: np.ndarray
    adj: np.ndarray
    mask: np.ndarray


def one_hot(value: object, choices: list[object]) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_NUMBERS)
        + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetFormalCharge()) / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass()) / 200.0,
        ]
    )


def graph_from_smiles(smiles: str, max_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = exp.mol_from_smiles(smiles)
    feature_dim = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
    x = np.zeros((max_atoms, feature_dim), dtype=np.float32)
    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=np.float32)
    if mol is None:
        return x, np.eye(max_atoms, dtype=np.float32), mask

    n_atoms = min(mol.GetNumAtoms(), max_atoms)
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        x[i] = np.asarray(atom_features(atom), dtype=np.float32)
        mask[i] = 1.0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj[:n_atoms, :n_atoms] += np.eye(n_atoms, dtype=np.float32)
    degree = adj.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    valid = degree > 0
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    return x, adj.astype(np.float32), mask


def build_graph_bundle(bundle: exp.MolBundle, max_atoms: int) -> GraphBundle:
    xs = []
    adjs = []
    masks = []
    for smi in bundle.smiles:
        x, adj, mask = graph_from_smiles(smi, max_atoms)
        xs.append(x)
        adjs.append(adj)
        masks.append(mask)
    return GraphBundle(
        x=np.stack(xs).astype(np.float32),
        adj=np.stack(adjs).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
    )


class GraphDataset(Dataset):
    def __init__(self, graphs: GraphBundle, y: np.ndarray, idx: np.ndarray):
        self.x = graphs.x[idx]
        self.adj = graphs.adj[idx]
        self.mask = graphs.mask[idx]
        self.y = y[idx].astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, item: int):
        return (
            torch.from_numpy(self.x[item]),
            torch.from_numpy(self.adj[item]),
            torch.from_numpy(self.mask[item]),
            torch.tensor(self.y[item], dtype=torch.float32),
        )


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_layers: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            propagated = torch.bmm(adj, h)
            update = torch.relu(layer(propagated))
            h = norm(h + self.dropout(update))
        mask3 = mask.unsqueeze(-1)
        pooled = (h * mask3).sum(dim=1) / mask3.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_gnn(
    train_graphs: GraphBundle,
    train_y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
) -> tuple[SimpleGCN, dict[str, float]]:
    set_seed(seed)
    train_sub, val_sub = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=seed + 313,
        stratify=train_y[train_idx],
    )
    train_ds = GraphDataset(train_graphs, train_y, train_sub)
    val_ds = GraphDataset(train_graphs, train_y, val_sub)
    loader = DataLoader(train_ds, batch_size=GNN_BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=GNN_BATCH_SIZE, shuffle=False)

    pos = float(np.sum(train_y[train_sub] == 1))
    neg = float(np.sum(train_y[train_sub] == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)

    def val_auc_for(model: SimpleGCN) -> float:
        probs = []
        labels = []
        model.eval()
        with torch.no_grad():
            for x, adj, mask, y in val_loader:
                x = x.to(DEVICE)
                adj = adj.to(DEVICE)
                mask = mask.to(DEVICE)
                logits = model(x, adj, mask)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.numpy())
        y_true = np.concatenate(labels).astype(int)
        prob = np.concatenate(probs).astype(float)
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(exp.roc_auc_score(y_true, prob))

    best_model = None
    best_config = None
    best_config_auc = -math.inf
    for config in GNN_CONFIGS:
        model = SimpleGCN(
            train_graphs.x.shape[-1],
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            num_layers=int(config["num_layers"]),
        ).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)

        best_state = None
        best_val_auc = -math.inf
        patience = 12
        stale = 0
        for epoch in range(GNN_EPOCHS):
            model.train()
            for x, adj, mask, y in loader:
                x = x.to(DEVICE)
                adj = adj.to(DEVICE)
                mask = mask.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x, adj, mask), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            current_auc = val_auc_for(model)
            if current_auc > best_val_auc + 1e-4:
                best_val_auc = current_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        if best_val_auc > best_config_auc:
            best_config_auc = best_val_auc
            best_model = model
            best_config = {
                "hidden_dim": float(config["hidden_dim"]),
                "dropout": float(config["dropout"]),
                "num_layers": float(config["num_layers"]),
                "validation_auroc": float(best_val_auc),
            }
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if best_model is None or best_config is None:
        raise RuntimeError("GNN sweep failed to select a model")
    best_model.eval()
    return best_model, best_config


def predict_gnn(model: SimpleGCN, graphs: GraphBundle, idx: np.ndarray) -> np.ndarray:
    dataset = GraphDataset(graphs, np.zeros(len(graphs.x), dtype=np.float32), idx)
    loader = DataLoader(dataset, batch_size=GNN_BATCH_SIZE, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for x, adj, mask, _ in loader:
            logits = model(x.to(DEVICE), adj.to(DEVICE), mask.to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs).astype(float)


def run_gnn_seed(
    seed: int,
    bbb: exp.MolBundle,
    b3db: exp.MolBundle,
    bbb_graphs: GraphBundle,
    b3db_graphs: GraphBundle,
    mondrian: bool,
    trained: tuple[SimpleGCN, dict[str, float]] | None = None,
) -> dict[str, float | None]:
    split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
    ext_split = exp.scaffold_split(b3db, seed + 101)
    external_idx_raw = ext_split["test"]
    external_idx = exp.exact_nonoverlap_indices(bbb.smiles, b3db, external_idx_raw)
    if len(external_idx) == 0 or len(np.unique(b3db.y[external_idx])) < 2:
        external_idx = external_idx_raw

    if trained is None:
        model, selected = train_gnn(bbb_graphs, bbb.y, split["train"], seed)
    else:
        model, selected = trained
    p_cal = predict_gnn(model, bbb_graphs, split["calibration"])
    p_test = predict_gnn(model, bbb_graphs, split["test"])
    p_external = predict_gnn(model, b3db_graphs, external_idx)

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
    out["gnn_epochs_max"] = float(GNN_EPOCHS)
    out["gnn_hidden_dim"] = float(selected["hidden_dim"])
    out["gnn_dropout"] = float(selected["dropout"])
    out["gnn_num_layers"] = float(selected["num_layers"])
    out["gnn_validation_auroc"] = float(selected["validation_auroc"])
    return {k: exp.safe_number(v) for k, v in out.items()}


def main() -> None:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    max_atoms = max(
        max((exp.mol_from_smiles(smi).GetNumAtoms() for smi in bbb.smiles if exp.mol_from_smiles(smi) is not None), default=1),
        max((exp.mol_from_smiles(smi).GetNumAtoms() for smi in b3db.smiles if exp.mol_from_smiles(smi) is not None), default=1),
    )
    print(f"advanced_baseline=device:{DEVICE} max_atoms={max_atoms}", flush=True)
    bbb_graphs = build_graph_bundle(bbb, max_atoms=max_atoms)
    b3db_graphs = build_graph_bundle(b3db, max_atoms=max_atoms)

    conditions = [
        ("gnn_message_passing_global_cp", False),
        ("gnn_message_passing_context_cp", True),
    ]
    output: dict[str, object] = {
        "status": "completed",
        "device": str(DEVICE),
        "model": "custom RDKit/PyTorch dense message-passing GCN with modest internal-validation architecture sweep",
        "hyperparameters": {
            "epochs_max": GNN_EPOCHS,
            "batch_size": GNN_BATCH_SIZE,
            "learning_rate": GNN_LR,
            "weight_decay": GNN_WEIGHT_DECAY,
            "architecture_grid": GNN_CONFIGS,
        },
        "conditions": {},
    }
    rows_by_condition: dict[str, list[dict[str, float | None]]] = {name: [] for name, _ in conditions}
    for seed in exp.SEEDS:
        trained = train_gnn(bbb_graphs, bbb.y, exp.duplicate_aware_stratified_three_way_split(bbb, seed)["train"], seed)
        for condition, mondrian in conditions:
            metrics = run_gnn_seed(seed, bbb, b3db, bbb_graphs, b3db_graphs, mondrian=mondrian, trained=trained)
            rows_by_condition[condition].append(metrics)
            exp.print_seed_metrics(condition, seed, metrics)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    for condition, _ in conditions:
        summary = exp.summarize(rows_by_condition[condition])
        exp.print_summaries(condition, summary)
        output["conditions"][condition] = {
            "seeds_run": len(rows_by_condition[condition]),
            "seed_metrics": rows_by_condition[condition],
            "summary": summary,
        }

    out_path = Path(__file__).resolve().parent / "advanced_baselines.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
