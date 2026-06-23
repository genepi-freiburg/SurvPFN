# SurvPFN: Towards Foundation Models for Survival Predictions

SurvPFN solves survival tasks zero-shot — without per-dataset fitting.

SurvPFN is a prior-data fitted network (PFN) for time-to-event survival prediction.
It's pretrained once on synthetic survival tasks generated
from structural causal models with Weibull event times and non-informative
censoring.
The model builds on [NanoTabPFN](https://github.com/automl/TFM-Playground) with
two survival specific additions: a two-routed target encoder for observed vs. censored
rows, and a right-censored negative log-likelihood as the training objective.

> 📄 **Published at the ICML 2026 Workshop on Foundation Models for Structured Data.**
> [Arxiv](https://arxiv.org/abs/2606.04564)

## Installation

Requires Python ≥ 3.12. Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Repository layout

```
src/prior/           # SCM-based Weibull survival prior generator
src/SurvPFN/         # model, losses, dataloader, training loop
```

## Generating the prior

```bash
python -m prior.generate src/prior/config/cfg_prior.yaml --output-dir priors/scm
```

This produces one HDF5 file per feature count (`scm_ncols_02.h5` … `scm_ncols_10.h5`),
plus a small held-out validation split from the same prior.

## Training

Example (headline `SurvPFN [NR]` configuration):

```bash
python -m SurvPFN.train \
    --loss native --rank \
    --file_pattern "priors/scm/scm_ncols_{ncols:02d}.h5" \
    --val_pattern  "priors/scm/val_scm_ncols_{ncols:02d}.h5"
```

Add `--ablation` to force the event-indicator feature column to `1.0` (model blind to censoring).


## Use on your own data

`SurvPFN` follows the [scikit-survival](https://scikit-survival.readthedocs.io)
estimator API — `fit` / `predict` / `predict_survival_function` with a
structured `y`. 

As this is a pre-trained model, `fit` just stores the support set (no gradient training); the prediction is zero-shot.

```python
import numpy as np
from sksurv.util import Surv
from huggingface_hub import hf_hub_download
from src.SurvPFN import SurvPFN

# Load your data
X_test, X_train, event_test, event_train, time_test, time_treain = ...

# Grab the pretrained checkpoint (or pass a local .pt path you trained yourself).
ckpt = hf_hub_download(repo_id="samuelboehm/SurvPFN", filename="survpfn_nr.pt")

# X: (n, d) float array; time: durations; event: bool (True = event, False = censored)
y_train = Surv.from_arrays(event=event_train, time=time_train)

est = SurvPFN(model_path=ckpt)
est.categorical_features_indices = [...]   # optional: indices of categorical columns

est.fit(X_train, y_train)
risk = est.predict(X_test)                       # higher = higher risk
surv_fns = est.predict_survival_function(X_test) # sksurv StepFunction per row
probs = np.stack([fn(times) for fn in surv_fns]) # evaluate at your own `times`
```

Keep `n_features` within the range the checkpoint was trained on (2–10 columns).
See `demo.py` for a full cross-validated evaluation example on SurvSet.

## Citation

If you use SurvPFN, please cite:

```bibtex
@article{boehm2026survpfn,
  title     = {SurvPFN: Towards Foundation Models for Survival Predictions},
  author    = {Böhm, Samuel and Purucker, Lennart and Hutter, Frank and Schlosser, Pascal},
  year      = {2026},
  journal={arXiv preprint arXiv:2606.04564},
}
```