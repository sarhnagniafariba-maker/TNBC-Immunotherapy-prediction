import argparse, yaml, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.gcn_model import GCNClassifier, predict_proba
from utils.graph_utils import build_block_adjacency
from utils.metrics import compute_metrics, bootstrap_ci

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

def infer_group_slices(cols, prefixes):
    idxs = []
    for pref in prefixes:
        grp = [i for i,c in enumerate(cols) if c.startswith(pref)]
        if len(grp) > 0:
            idxs.append(slice(min(grp), max(grp)+1))
    return idxs

def main(cfg, model_path):
    device = torch.device("cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")
    dcfg = cfg["data"]
    val_ds = CSVDataset(dcfg["val_path"], dcfg["target_col"], dcfg.get("id_col"))
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # load/build graph
    gcfg = cfg["graph"]
    if os.path.exists(gcfg["path"]):
        import numpy as np
        A_hat = np.load(gcfg["path"])
    else:
        prefixes = [g["prefix"] for g in dcfg["feature_groups"]]
        A_hat = build_block_adjacency(val_ds.feat_cols, prefixes, intra_weight=1.0, inter_weight=0.05)
        os.makedirs(os.path.dirname(gcfg["path"]), exist_ok=True)
        np.save(gcfg["path"], A_hat)
    A_hat = torch.tensor(A_hat, dtype=torch.float32, device=device)

    prefixes = [g["prefix"] for g in dcfg["feature_groups"]]
    group_slices = infer_group_slices(val_ds.feat_cols, prefixes)

    model = GCNClassifier(in_features=val_ds.X.shape[1],
                          A_hat=A_hat,
                          hidden_dims=tuple(cfg["model"]["hidden_dims"]),
                          dropout=cfg["model"]["dropout"],
                          group_slices=group_slices if cfg["model"]["use_attention"] else None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            logit, _ = model(X)
            p = torch.sigmoid(logit).cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    m = compute_metrics(labels, probs, threshold=cfg["eval"]["threshold"])

    # bootstrap CIs
    auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(lambda yt, yp: __import__("sklearn.metrics").metrics.roc_auc_score(yt, yp),
                                                  labels, probs, n_boot=1000, alpha=0.05)
    aupr_mean, aupr_lo, aupr_hi = bootstrap_ci(lambda yt, yp: __import__("sklearn.metrics").metrics.average_precision_score(yt, yp),
                                               labels, probs, n_boot=1000, alpha=0.05)

    m_out = {
        **m,
        "auroc_ci": [auroc_lo, auroc_hi],
        "aupr_ci": [aupr_lo, aupr_hi]
    }

    print("Metrics:", m_out)
    out_path = os.path.join(cfg["save"]["dir"], "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(m_out, f, indent=2)
    print("Saved:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.model)