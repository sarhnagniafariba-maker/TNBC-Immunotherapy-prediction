import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "aupr": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

def bootstrap_ci(func, y_true, y_prob, n_boot=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    stats = []
    n = len(y_true)
    idx = np.arange(n)
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        stats.append(func(y_true[b], y_prob[b]))
    lo = np.percentile(stats, 100*alpha/2)
    hi = np.percentile(stats, 100*(1 - alpha/2))
    return float(np.mean(stats)), float(lo), float(hi)