"""Evaluate a SurvPFN checkpoint on SurvSet.

Usage:
    python demo.py --model_path path/to/final.pt
"""

import argparse
import warnings
import numpy as np
from huggingface_hub import hf_hub_download
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from SurvivalEVAL import SurvivalEvaluator

from src.SurvPFN import SurvPFN
from src.SurvPFN.dataloader import (
    list_survset_datasets,
    load_survset_dataset,
    make_cv_splits,
)

# Default checkpoint pulled from the Hugging Face Hub when --model_path is omitted.
HF_REPO_ID = "samuelboehm/SurvPFN"
HF_FILENAME = "survpfn_nr.pt"


def evaluate_fold(est, X_tr, X_te, t_tr, e_tr, t_te, e_te, n_eval=50):
    if e_tr.sum() < 3 or not e_te.any():
        return None

    eps = 1e-6
    t_lo = float(max(t_tr.min(), t_te.min())) + eps
    t_hi = float(min(t_tr.max(), t_te.max())) - eps
    if t_lo >= t_hi:
        return None
    eval_times = np.linspace(t_lo, t_hi, n_eval)

    y_tr = Surv.from_arrays(event=e_tr, time=t_tr)
    y_te = Surv.from_arrays(event=e_te, time=t_te)

    est.fit(X_tr, y_tr)
    risk = est.predict(X_te)
    sfns = est.predict_survival_function(X_te)
    surv_probs = np.stack([fn(eval_times) for fn in sfns])

    try:
        ci = float(concordance_index_censored(e_te, t_te, risk)[0])
    except Exception:
        ci = float('nan')

    try:
        ibs = float(integrated_brier_score(y_tr, y_te, surv_probs, eval_times))
    except Exception:
        ibs = float('nan')

    ici = float('nan')
    try:
        et_full = np.concatenate([[0.0], eval_times])
        sp_full = np.concatenate(
            [np.ones((surv_probs.shape[0], 1)), surv_probs], axis=1,
        )
        evaluator = SurvivalEvaluator(
            pred_survs=sp_full, time_coordinates=et_full,
            event_times=t_te, event_indicators=e_te.astype(int),
            train_event_times=t_tr, train_event_indicators=e_tr.astype(int),
        )
        target_time = float(np.clip(
            np.median(np.concatenate([t_tr, t_te])), t_lo, t_hi,
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = evaluator.integrated_calibration_index(
                target_time=target_time, knots=3, draw_figure=False,
            )
        ici = float(summary.get('ICI', float('nan')))
    except Exception:
        pass

    return ci, ibs, ici


def main():
    parser = argparse.ArgumentParser(description="SurvPFN SurvSet demo.")
    parser.add_argument("--model_path", default=None,
                        help="Path to a SurvPFN .pt artifact. If omitted, the "
                             f"default checkpoint is downloaded from the "
                             f"Hugging Face Hub ({HF_REPO_ID}).")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, or None for auto.")
    parser.add_argument("--min_rows", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--min_feat", type=int, default=2)
    parser.add_argument("--max_feat", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds_list = list_survset_datasets(
        min_rows=args.min_rows, max_rows=args.max_rows,
        min_features=args.min_feat, max_features=args.max_feat,
        exclude_td=True,
    )
    print(f"Found {len(ds_list)} SurvSet datasets fitting the model shape")

    model_path = args.model_path
    if model_path is None:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        print(f"Downloaded checkpoint from HF: {HF_REPO_ID}/{HF_FILENAME}")

    est = SurvPFN(model_path=model_path, device=args.device)

    header = f"{'dataset':<24s} {'C-index':>9s} {'IBS':>9s} {'ICI':>9s}"
    print()
    print(header)
    print("-" * len(header))

    all_ci, all_ibs, all_ici = [], [], []

    for ds_name, _, _ in ds_list:
        try:
            X_all, T_all, E_all, is_cat = load_survset_dataset(ds_name)
        except Exception:
            continue

        est.categorical_features_indices = np.where(is_cat)[0].tolist() or None

        try:
            splits = make_cv_splits(E_all, n_folds=args.n_folds,
                                    random_state=args.seed)
        except Exception:
            continue

        fold_ci, fold_ibs, fold_ici = [], [], []
        for tr_idx, te_idx in splits:
            res = evaluate_fold(
                est,
                X_all[tr_idx], X_all[te_idx],
                T_all[tr_idx].astype(float), E_all[tr_idx].astype(bool),
                T_all[te_idx].astype(float), E_all[te_idx].astype(bool),
            )
            if res is None:
                continue
            ci, ibs, ici = res
            if not np.isnan(ci):  fold_ci.append(ci)
            if not np.isnan(ibs): fold_ibs.append(ibs)
            if not np.isnan(ici): fold_ici.append(ici)

        if not fold_ci:
            continue
        ci_m  = float(np.mean(fold_ci))
        ibs_m = float(np.mean(fold_ibs)) if fold_ibs else float('nan')
        ici_m = float(np.mean(fold_ici)) if fold_ici else float('nan')
        print(f"{ds_name:<24s} {ci_m:>9.4f} {ibs_m:>9.4f} {ici_m:>9.4f}")
        all_ci.append(ci_m)
        all_ibs.append(ibs_m)
        all_ici.append(ici_m)

    print("-" * len(header))
    if all_ci:
        print(f"{'macro mean':<24s} "
              f"{np.mean(all_ci):>9.4f} "
              f"{np.nanmean(all_ibs):>9.4f} "
              f"{np.nanmean(all_ici):>9.4f}")
    else:
        print("no datasets produced valid metrics")


if __name__ == "__main__":
    main()
