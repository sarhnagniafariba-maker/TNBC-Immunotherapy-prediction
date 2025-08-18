import torch
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import yaml
import os
from ..models.gcn_model import GCNModel

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

def preprocess_fold(X_train, X_test, y_train):
    """Preprocessing inside fold to avoid leakage."""
    if config['preprocessing']['imputation'] == 'knn':
        imputer = KNNImputer(n_neighbors=config['preprocessing']['knn_k'])
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    
    var_thresh = VarianceThreshold(threshold=config['preprocessing']['variance_threshold'])
    X_train = var_thresh.fit_transform(X_train)
    X_test = var_thresh.transform(X_test)
    
    selector = SelectKBest(f_classif, k='all')
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    
    enet = ElasticNet(alpha=config['preprocessing']['elastic_net_alpha'], l1_ratio=config['preprocessing']['elastic_net_l1_ratio'])
    enet.fit(X_train, y_train)
    important = np.abs(enet.coef_) > 0
    X_train = X_train[:, important]
    X_test = X_test[:, important]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

def train():
    torch.manual_seed(config['model']['seed'])
    X, y = load_data()
    adj = np.load(config['data']['graph'])
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    
    outer_cv = StratifiedKFold(n_splits=config['training']['nested_cv_folds'])
    aucs = []
    
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        
        inner_cv = StratifiedKFold(n_splits=config['training']['nested_cv_folds'])
        best_auc = 0
        best_model = None
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
            X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
            y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
            
            X_inner_train, X_inner_val = preprocess_fold(X_inner_train, X_inner_val, y_inner_train)
            
            data_train = Data(x=torch.tensor(X_inner_train, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y_inner_train, dtype=torch.float))
            data_val = Data(x=torch.tensor(X_inner_val, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y_inner_val, dtype=torch.float))
            
            model = GCNModel(in_channels=X_inner_train.shape[1], hidden_channels=config['model']['gcn_layers'], dropout=config['model']['dropout'])
            optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['model']['l2_reg'])
            criterion = nn.BCELoss()
            
            patience = config['training']['early_stop_patience']
            min_loss = float('inf')
            counter = 0
            
            for epoch in range(config['training']['epochs']):
                model.train()
                optimizer.zero_grad()
                out, _ = model(data_train.x, data_train.edge_index)
                loss = criterion(out.squeeze(), data_train.y)
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_out, _ = model(data_val.x, data_val.edge_index)
                    val_loss = criterion(val_out.squeeze(), data_val.y)
                
                if val_loss < min_loss:
                    min_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    break
            
            with torch.no_grad():
                val_pred = model(data_val.x, data_val.edge_index)[0].squeeze().numpy()
            inner_auc = roc_auc_score(y_inner_val, val_pred)
            if inner_auc > best_auc:
                best_auc = inner_auc
                best_model = model
        
        X_outer_train, X_outer_test = preprocess_fold(X_outer_train, X_outer_test, y_outer_train)
        data_test = Data(x=torch.tensor(X_outer_test, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y_outer_test, dtype=torch.float))
        
        with torch.no_grad():
            test_pred, _ = best_model(data_test.x, data_test.edge_index)
            test_pred = test_pred.squeeze().numpy()
        auc = roc_auc_score(y_outer_test, test_pred)
        aucs.append(auc)
    
    os.makedirs('../results', exist_ok=True)
    torch.save(best_model.state_dict(), config['output']['model_path'])
    print(f"Mean AUC: {np.mean(aucs):.2f} ± {np.std(aucs):.2f}")

if __name__ == "__main__":
    train()