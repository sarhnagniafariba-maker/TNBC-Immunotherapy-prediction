import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A_hat):
        # x: (B, F)  A_hat: (F, F)
        # Kipf: H = A_hat @ X @ W
        H = torch.matmul(x, A_hat.T)      # (B, F)
        H = self.lin(H)                   # (B, out_dim)
        H = self.act(H)
        return self.drop(H)

class AttentionGroupPool(nn.Module):
    def __init__(self, hidden_dim, group_slices):
        super().__init__()
        self.group_slices = group_slices
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, H):
        # H: (B, F') features after GCN (we treat columns as features); for pooling, we average within groups,
        # then compute an attention weight per group and return weighted sum.
        pools = []
        for sl in self.group_slices:
            pools.append(H[:, sl].mean(dim=1, keepdim=True))  # (B,1)
        G = torch.cat(pools, dim=1)  # (B, G)
        # attention over groups
        a = torch.softmax(self.score(G.unsqueeze(-1)).squeeze(-1), dim=1)  # (B, G)
        Z = (a * G).sum(dim=1)  # (B,)
        return Z, a

class GCNClassifier(nn.Module):
    def __init__(self, in_features, A_hat, hidden_dims=(256,128), dropout=0.2, group_slices=None):
        super().__init__()
        self.register_buffer('A_hat', A_hat)   # (F,F)
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(GCNLayer(prev, h, dropout=dropout))
            prev = h
        self.gcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(prev)
        self.group_slices = group_slices
        if group_slices is not None and len(group_slices) > 0:
            self.pool = AttentionGroupPool(prev, group_slices)
            clf_in = 1  # pooled scalar per sample
        else:
            self.pool = None
            clf_in = prev
        self.cls = nn.Sequential(
            nn.Linear(clf_in, max(16, clf_in)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, clf_in), 1)
        )

    def forward(self, x):
        # x: (B, F)
        H = self.gcn[0](x, self.A_hat)
        for layer in list(self.gcn)[1:]:
            H = layer(H, self.A_hat) if isinstance(layer, GCNLayer) else layer(H)
        H = self.norm(H)
        if self.pool is not None:
            Z, attn = self.pool(H)   # (B,), (B,G)
            logit = self.cls(Z.unsqueeze(1)).squeeze(1)
            return logit, attn
        else:
            logit = self.cls(H)
            return logit.squeeze(-1), None

def predict_proba(logit):
    return torch.sigmoid(logit)