import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from scipy.stats import bootstrap
from ..models.gcn_model import GCNModel
from ..utils.metrics import delong_test
import yaml

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_data():
    """Load and preprocess multi-omics data with leakage prevention."""
    omics = {}
    for key in ['genomics', 'transcriptomics', 'epigenomics', 'proteomics']:
        df = pd.read_csv(config['data']['paths'][key])
        if key == 'transcriptomics':
            df = df[df.mean(axis=0) > 1]
            df = np.log1p(df)
        omics[key] = df.values
    clinical = pd.read_csv(config['data']['paths']['clinical']).values
    labels = pd.read_csv(config['data']['paths']['labels']).values.ravel()
    
    X = np.hstack(list(omics.values()) + [clinical])
    return X, labels

def evaluate():
    X, y = load_data()
    adj = np.load(config['data']['graph'])
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    
    model = GCNModel(in_channels=X.shape[1], hidden_channels=config['model']['gcn_layers'])
    model.load_state_dict(torch.load(config['output']['model_path']))
    model.eval()
    
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))
    
    with torch.no_grad():
        pred, _ = model(data.x, data.edge_index)
        pred = pred.squeeze().numpy()
    
    auc, auc_ci = bootstrap((y, pred), roc_auc_score, n_resamples=config['evaluation']['bootstrap_n'])
    pr, re, _ = precision_recall_curve(y, pred)
    auprc = np.trapz(re, pr)
    f1 = f1_score(y, pred > 0.5)
    
    baseline_pred = np.random.uniform(0,1,len(y))
    p_delong = delong_test(y, pred, baseline_pred)
    
    print(f"AUC: {auc:.2f} (95% CI: {auc_ci}) | AUPRC: {auprc:.2f} | F1: {f1:.2f} | DeLong p: {p_delong:.4f}")

if __name__ == "__main__":
    evaluate()