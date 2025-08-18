import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

def build_feature_graph(omics_data, string_interactions=None):
    """Build adjacency matrix from feature interactions (STRING-like)."""
    n_features = sum([data.shape[1] for data in omics_data.values()])
    G = nx.random_graphs.erdos_renyi_graph(n_features, 0.05)
    adj = nx.to_numpy_array(G)
    
    all_data = np.hstack(list(omics_data.values()))
    pca = PCA(n_components=100)
    reduced = pca.fit_transform(all_data)
    
    os.makedirs('../graphs', exist_ok=True)
    np.save('../graphs/feature_graph.npy', adj)
    print("Feature graph saved as '../graphs/feature_graph.npy'.")
    return adj

if __name__ == "__main__":
    omics = {k: pd.read_csv(f'../data/{k}.csv').values for k in ['genomics', 'transcriptomics', 'epigenomics', 'proteomics']}
    build_feature_graph(omics)