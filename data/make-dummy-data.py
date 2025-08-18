import numpy as np
import pandas as pd
import os
import yaml

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def generate_dummy_data(n_samples=450, n_features=1000):
    """Generate synthetic multi-omics data for TNBC-like cohorts."""
    omics = {}
    omics['genomics'] = np.random.binomial(1, 0.1, size=(n_samples, n_features // 4))
    omics['transcriptomics'] = np.random.lognormal(5, 2, size=(n_samples, n_features // 4))
    omics['epigenomics'] = np.random.beta(0.5, 0.5, size=(n_samples, n_features // 4))
    omics['proteomics'] = np.random.normal(0, 1, size=(n_samples, n_features // 4))
    
    clinical = pd.DataFrame({
        'age': np.random.randint(30, 70, n_samples),
        'stage': np.random.choice([1,2,3,4], n_samples),
        'prior_therapy': np.random.binomial(1, 0.5, n_samples)
    })
    
    labels = np.random.binomial(1, 0.3, n_samples)
    
    os.makedirs('../data', exist_ok=True)
    for key, data in omics.items():
        pd.DataFrame(data).to_csv(f'../data/{key}.csv', index=False)
    clinical.to_csv('../data/clinical.csv', index=False)
    pd.Series(labels).to_csv('../data/labels.csv', index=False)
    
    print("Dummy data generated and saved in '../data/' folder.")

if __name__ == "__main__":
    if config['data']['dummy']:
        generate_dummy_data()