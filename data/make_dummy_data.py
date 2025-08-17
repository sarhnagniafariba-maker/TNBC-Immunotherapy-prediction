import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.graph_utils import build_block_adjacency
import os

def make_block(n, p, prefix, scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, scale, size=(n, p))
    cols = [f"{prefix}{i:03d}" for i in range(p)]
    return pd.DataFrame(X, columns=cols)

def main():
    n = 500
    gen = make_block(n, 60, "gen_", 1.0, 1)
    tx  = make_block(n, 60, "tx_",  1.0, 2)
    epi = make_block(n, 60, "epi_", 1.0, 3)
    prot= make_block(n, 40, "prot_",1.0, 4)
    clin= make_block(n, 10, "clin_",1.0, 5)
    sc  = make_block(n, 20, "sc_",  1.0, 6)
    sp  = make_block(n, 20, "sp_",  1.0, 7)

    # synthetic label signal using a few nonlinear interactions
    signal = (gen["gen_000"]*0.7 - tx["tx_005"]*0.6 + epi["epi_010"]*0.5
              + np.tanh(gen["gen_007"])*0.3 + (tx["tx_012"]**2)*0.1
              + prot["prot_001"]*0.4 + clin["clin_003"]*0.2 - sc["sc_004"]*0.3 + sp["sp_002"]*0.25)
    prob = 1/(1 + np.exp(-signal.values))
    rng = np.random.default_rng(123)
    y = (rng.uniform(0,1,size=len(prob)) < prob).astype(int)

    df = pd.concat([gen, tx, epi, prot, clin, sc, sp], axis=1)
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["label"] = y

    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label"])
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    print("Wrote data/train.csv and data/val.csv")

    # Build and save feature adjacency
    prefixes = ["gen_","tx_","epi_","prot_","clin_","sc_","sp_"]
    A_hat = build_block_adjacency([c for c in df.columns if c not in ["label","sample_id"]], prefixes,
                                  intra_weight=1.0, inter_weight=0.05)
    os.makedirs("graphs", exist_ok=True)
    import numpy as np
    np.save("graphs/feature_graph.npy", A_hat)
    print("Wrote graphs/feature_graph.npy")

if __name__ == "__main__":
    main()