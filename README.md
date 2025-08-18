# An Explainable Multi-Omics Deep Learning Framework for Precision Prediction of Immunotherapy Response in Triple-Negative Breast Cancer

**Date**: 26 Mordad 1404 (17 August 2025)

This repository contains the implementation for our manuscript describing an explainable multi-omics framework to predict immunotherapy response in triple-negative breast cancer (TNBC). It integrates genomic, transcriptomic, epigenomic, proteomic, clinical, and features from single-cell and spatial transcriptomics. The core model is a Graph Convolutional Network (GCN) with attention, supporting SHAP explanations.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/make_dummy_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/explain.py