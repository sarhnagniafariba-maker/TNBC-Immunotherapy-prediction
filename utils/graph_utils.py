import numpy as np

def build_block_adjacency(feature_names, group_prefixes, intra_weight=1.0, inter_weight=0.05):
    """Build a simple block adjacency matrix based on feature-name prefixes.
    - feature_names: list[str]
    - group_prefixes: list[str], e.g., ["gen_", "tx_", ...]
    Returns: A (F x F) numpy array, symmetrized with self-loops normalized (A_hat).
    """
    F = len(feature_names)
    A = np.full((F, F), inter_weight, dtype=np.float32)
    # assign group ids
    gid = np.zeros(F, dtype=int)
    for i, name in enumerate(feature_names):
        for g, pref in enumerate(group_prefixes):
            if name.startswith(pref):
                gid[i] = g
                break
    # strengthen intra-group connections
    for i in range(F):
        for j in range(F):
            if gid[i] == gid[j]:
                A[i, j] = intra_weight
    # add self loops
    np.fill_diagonal(A, intra_weight + 1.0)
    # symmetric normalization A_hat = D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.astype(np.float32)