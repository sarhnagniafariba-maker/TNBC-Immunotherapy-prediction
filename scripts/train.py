import argparse, yaml, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.graph_utils import build_block_adjacency
from models.gcn_model import GCNClassifier, predict_proba

class CSVDataset(Dataset):
    def __init__(self, path, target_col, id_col=None):
        df = pd.read_csv(path)
        self.ids = df[id_col].values if id_col in df.columns else np.arange(len(df))
        self.y = df[target_col].values.astype(np.float32)
        feat_cols = [c for c in df.columns if c not in [target_col, id_col]]
        self.feat_cols = feat_cols
        self.X = df[feat_cols].values.astype(np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx]), self.ids[idx]

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def infer_group_slices(cols, prefixes):
    idxs = []
    for pref in prefixes:
        grp = [i for i,c in enumerate(cols) if c.startswith(pref)]
        if len(grp) > 0:
            idxs.append(slice(min(grp), max(grp)+1))
    return idxs

def main(cfg):
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")

    dcfg = cfg["data"]
    train_ds = CSVDataset(dcfg["train_path"], dcfg["target_col"], dcfg.get("id_col"))
    val_ds   = CSVDataset(dcfg["val_path"], dcfg["target_col"], dcfg.get("id_col"))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    # Build or load feature graph adjacency
    gcfg = cfg["graph"]
    if os.path.exists(gcfg["path"]):
        A_hat = np.load(gcfg["path"])
    else:
        prefixes = [g["prefix"] for g in dcfg["feature_groups"]]
        A_hat = build_block_adjacency(train_ds.feat_cols, prefixes, intra_weight=1.0, inter_weight=0.05)
        os.makedirs(os.path.dirname(gcfg["path"]), exist_ok=True)
        np.save(gcfg["path"], A_hat)
    A_hat = torch.tensor(A_hat, dtype=torch.float32, device=device)

    # Group slices for attention pooling
    prefixes = [g["prefix"] for g in dcfg["feature_groups"]]
    group_slices = infer_group_slices(train_ds.feat_cols, prefixes)

    model = GCNClassifier(in_features=train_ds.X.shape[1],
                          A_hat=A_hat,
                          hidden_dims=tuple(cfg["model"]["hidden_dims"]),
                          dropout=cfg["model"]["dropout"],
                          group_slices=group_slices if cfg["model"]["use_attention"] else None).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    os.makedirs(cfg["save"]["dir"], exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        tr_loss = 0.0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logit, _ = model(X)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                logit, _ = model(X)
                loss = loss_fn(logit, y)
                vl_loss += loss.item() * X.size(0)
        vl_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f}")
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), os.path.join(cfg["save"]["dir"], cfg["save"]["model_name"]))

    print("Training completed. Best val loss:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)