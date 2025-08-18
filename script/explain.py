import torch
import shap
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_rate
from ..models.gcn_model import GCNModel
from ..utils.metrics import roc_auc_score
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

def explain():
    X, y = load_data()
    adj = np.load(config['data']['graph'])
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    
    model = GCNModel(in_channels=X.shape[1], hidden_channels=config['model']['gcn_layers'])
    model.load_state_dict(torch.load(config['output']['model_path']))
    model.eval()
    
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index)
    
    _, attn_weights = model(data.x, data.edge_index)
    attn_weights = attn_weights.detach().numpy()
    
    explainer = shap.DeepExplainer(model, data.x[:config['explainability']['shap_samples']])
    shap_values = explainer.shap_values(data.x)
    
    pathway_scores = np.mean(shap_values, axis=0)
    q_values = false_discovery_rate(pathway_scores)
    
    ablations = {}
    n_features_per_omic = X.shape[1] // 4
    for i, omic in enumerate(['genomics', 'transcriptomics', 'epigenomics', 'proteomics']):
        X_ablated = X.copy()
        start = i * n_features_per_omic
        end = start + n_features_per_omic
        X_ablated[:, start:end] = 0
        data_ablated = Data(x=torch.tensor(X_ablated, dtype=torch.float), edge_index=edge_index)
        with torch.no_grad():
            pred_ablated, _ = model(data_ablated.x, data_ablated.edge_index)
        auc_ablated = roc_auc_score(y, pred_ablated.squeeze().numpy())
        ablations[omic] = auc_ablated
    
    os.makedirs('../results', exist_ok=True)
    pd.to_pickle({'attn': attn_weights, 'shap': shap_values, 'q_values': q_values, 'ablations': ablations}, config['output']['explanations'])
    print("Explanations saved.")

if __name__ == "__main__":
    explain()