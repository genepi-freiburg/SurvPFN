# Survival prior

Generates synthetic survival datasets to pretrain SurvPFN. Each dataset is drawn
from a randomly-sampled **Structural Causal Model (SCM)**  whose node mechanisms (MLP / XGBoost)
produce correlated covariates and the risk score. Event times are generated using a Weibull distribution
from the risk scroe. Censoring is applied on top.

## Pipeline

1. **Sample an SCM** (`scm/`) — random DAG + node mechanisms + noise.
2. **Propagate** to get node values. The last 1–2 topological nodes are reserved
   as outcomes; the rest are candidate features. Collapsed/exploding nodes are
   re-sampled until variance is acceptable.
3. **Pick features** from the outcome's ancestors, optionally categorize some.
4. **Map outcomes → event times** (`survival.py`)
5. **Censor** (`apply_censoring_adaptive`) quantile-based administrative
   censoring with staggered enrollment + random loss-to-follow-up.

## Usage

Batch generation to HDF5:

```bash
python -m prior.generate config/cfg_prior.yaml --output-dir priors/scm
```

This writes one `*_ncols_NN.h5` file per feature count in
`[n_features, n_features_max]`, each holding `n_datasets` datasets.
