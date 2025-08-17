import argparse, yaml, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.gcn_model import GCNClassifier

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

def main(cfg, model_path, n=32):
    device = torch.device("cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")
    dcfg = cfg["data"]
    ds = CSVDataset(dcfg["val_path"], dcfg["target_col"], dcfg.get("id_col"))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    import numpy as np
    A_hat = np.load(cfg["graph"]["path"])
    A_hat = torch.tensor(A_hat, dtype=torch.float32, device=device)

    prefixes = [g["prefix"] for g in dcfg["feature_groups"]]
    # simple slices heuristic
    def infer_group_slices(cols, prefixes):
        idxs = []
        for pref in prefixes:
            grp = [i for i,c in enumerate(cols) if c.startswith(pref)]
            if len(grp) > 0:
                idxs.append(slice(min(grp), max(grp)+1))
        return idxs
    group_slices = infer_group_slices(ds.feat_cols, prefixes)

    model = GCNClassifier(in_features=ds.X.shape[1],
                          A_hat=A_hat,
                          hidden_dims=tuple(cfg["model"]["hidden_dims"]),
                          dropout=cfg["model"]["dropout"],
                          group_slices=group_slices if cfg["model"]["use_attention"] else None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    attributions = []
    ids = []
    for k, (X, _, sid) in enumerate(loader):
        if k >= n: break
        X = X.to(device).requires_grad_(True)
        logit, _ = model(X)
        prob = torch.sigmoid(logit).mean()
        prob.backward()
        grad = X.grad.detach().cpu().numpy()[0]
        attr = (grad * X.detach().cpu().numpy()[0])  # grad x input
        attributions.append(attr.tolist())
        ids.append(sid if isinstance(sid, str) else str(sid.item()))

    # save top features per sample
    feat_cols = ds.feat_cols
    report = []
    for sid, attr in zip(ids, attributions):
        arr = np.array(attr)
        top_idx = np.argsort(-np.abs(arr))[:10]
        report.append({
            "sample_id": sid,
            "top10_features": [feat_cols[i] for i in top_idx],
            "top10_scores": [float(arr[i]) for i in top_idx]
        })

    out_path = os.path.join(cfg["save"]["dir"], "explain_top10.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.model, n=args.n)