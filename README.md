# P2-ETF-HRP-ALLOCATOR

**Hierarchical Risk Parity (HRP) Allocation Engine for ETF Portfolios**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-HRP-ALLOCATOR/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HRP-ALLOCATOR/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--hrp--allocator--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hrp-allocator-results)

## Overview

`P2-ETF-HRP-ALLOCATOR` generates robust, risk-based portfolio weights for ETFs using **Hierarchical Risk Parity (HRP)** . Unlike traditional mean-variance optimization, HRP does not require expected return estimates, making it significantly more stable and robust to estimation error.

The engine uses a three-step process:
1. **Hierarchical Clustering** – Groups ETFs based on their correlation structure.
2. **Quasi-Diagonalization** – Reorders assets to align with the cluster hierarchy.
3. **Recursive Bisection** – Allocates risk equally across the hierarchy using inverse-variance weights.

Results are pushed daily to a dedicated Hugging Face dataset and visualized via a Streamlit dashboard.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data is sourced from: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology

### Hierarchical Risk Parity (HRP)

1. **Covariance Estimation:** Compute the covariance matrix from the last 252 trading days of log returns.
2. **Distance Matrix:** Convert correlation to distance: `d = sqrt((1 - ρ) / 2)`.
3. **Hierarchical Clustering:** Apply Ward's linkage to form a cluster tree.
4. **Quasi-Diagonalization:** Reorder assets based on cluster leaves.
5. **Recursive Bisection:** Split clusters and allocate risk proportionally to inverse cluster variance.

## File Structure
P2-ETF-HRP-ALLOCATOR/
├── config.py # Paths, universes, HRP parameters
├── data_manager.py # Data loading and preprocessing
├── hrp_model.py # Core HRP allocation logic
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-HRP-ALLOCATOR.git
cd P2-ETF-HRP-ALLOCATOR
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Pie & Bar Charts: Visualize portfolio weights across ETFs.

Weight Table: Detailed allocation percentages.

Cluster Dendrogram: Understand how assets are grouped by correlation.

Three Tabs: Separate views for Combined, Equity, and FI/Commodities universes.

License
MIT License
