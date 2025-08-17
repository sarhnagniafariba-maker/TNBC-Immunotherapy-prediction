# An Explainable Multi-Omics Deep Learning Framework for Precision Prediction of Immunotherapy Response in Triple-Negative Breast Cancer

This repository contains the **implementation scaffold** for our manuscript (in preparation for submission) describing an explainable multi-omics framework to predict immunotherapy response in **triple‑negative breast cancer (TNBC)**.

It integrates **genomic, transcriptomic, epigenomic, proteomic, clinical**, and summary features from **single‑cell** and **spatial transcriptomics**. The core model applies a lightweight **Graph Convolutional Network (GCN)** over a feature‑graph with an optional **attention pooling** module, and supports **SHAP** explanations.

> This repository is prepared for academic collaboration and code review by prospective co‑authors and PIs. No patient‑level data are included. A synthetic dataset and feature‑graph generator are provided to run the pipeline end‑to‑end.

---

## Quick start

```bash
# (optional) create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# generate synthetic TNBC-like multi-omics data and a feature graph
python data/make_dummy_data.py

# train (on dummy data)
python scripts/train.py --config config.yaml

# evaluate
python scripts/evaluate.py --model results/model.pt --config config.yaml

# (optional) get simple explanations (gradient x input)
python scripts/explain.py --model results/model.pt --config config.yaml --n 32
```

---

## Repository structure
```
├── data/
│   ├── make_dummy_data.py   # synthetic multi-omics generator (no PHI)
│   └── README.md            # data handling notes
├── graphs/
│   └── feature_graph.npy    # adjacency (F x F) built from feature groups
├── models/
│   └── gcn_model.py         # minimal GCN + attention over feature groups
├── scripts/
│   ├── train.py             # training loop
│   ├── evaluate.py          # evaluation & metrics
│   └── explain.py           # quick attribution demo (grad x input)
├── utils/
│   ├── metrics.py           # AUROC, AUPR, Accuracy (+ bootstrap CIs)
│   └── graph_utils.py       # build adjacency from feature groups
├── results/                 # outputs (gitignored except .gitkeep)
├── config.yaml              # configuration
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

---

## Notes on rigor & reproducibility
- **No data leakage:** Feature filtering and any normalization must be fit on training folds only.
- **Validation:** Supports nested CV in future updates; current scaffold includes hold‑out evaluation and bootstrap CIs.
- **Statistics:** Report AUC/AUPRC/Accuracy with 95% CIs (bootstrap). For ROC comparisons, use DeLong’s test (not included here by default).
- **Explainability:** Attention weights (group‑level) and simple attributions (grad×input); add SHAP for deeper analysis if needed.

---

## Citation
If you find this repository useful, please cite our manuscript (pre‑submission). A preprint DOI will be added here when available.

---

## Contact
- **Name:** Fariba Sarhnagnia
- **Email:** sarhnagniafariba@gmail.com
- **Affiliation:** [Your Institution]
```)