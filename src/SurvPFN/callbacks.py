"""Validation callbacks for SurvPFN training."""

import warnings
import numpy as np
import torch
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
)
from sksurv.util import Surv
from tfmplayground.callbacks import Callback
from pfns.bar_distribution import FullSupportBarDistribution
from SurvivalEVAL import SurvivalEvaluator
import wandb

from ._artifact import save_artifact
from .config import SurvPFNConfig
from .survival import SurvPFN


def _build_est_from_batch(model, batch, sep, config, bucket_edges, device):
    """Extract train/test split from a prior batch (B=1) and return a fitted
    SurvPFN plus the held-out test arrays.

    Returns (est, X_te, t_te, e_te, t_tr, e_tr).
    """
    X = batch['x'][0].cpu().numpy().astype(np.float32)
    t = batch['observed_time'][0].cpu().numpy().astype(float)
    e = batch['event_indicator'][0].cpu().numpy().astype(bool)

    is_cat_t = batch.get('is_categorical')
    cat_indices = None
    if is_cat_t is not None:
        flat = is_cat_t.cpu().numpy().reshape(-1)
        is_cat = flat[: X.shape[1]].astype(bool)
        cat_indices = np.where(is_cat)[0].tolist() or None

    est = SurvPFN.from_live_model(
        model=model, config=config, bucket_edges=bucket_edges,
        device=device, categorical_features_indices=cat_indices,
    )
    y_tr = Surv.from_arrays(event=e[:sep], time=t[:sep])
    est.fit(X[:sep], y_tr)
    return est, X[sep:], t[sep:], e[sep:], t[:sep], e[:sep]


class ValidationLoggingCallback(Callback):
    """Validation metrics for the ICML camera-ready (denormalised space).

    Reports the three survival metrics from the paper, averaged across the
    validation batches:
      - ci_uno  Uno's IPCW C-index (sksurv)
      - ibs     IPCW-weighted Integrated Brier Score (SurvivalEVAL)
      - ici     Integrated Calibration Index at median follow-up (SurvivalEVAL)
    """

    def __init__(self, val_loader, device, dist: FullSupportBarDistribution,
                 config: SurvPFNConfig | None = None,
                 n_eval_times: int = 50):
        self.val_loader = val_loader
        self.device = device
        self.dist = dist
        self.config = config
        self.n_eval_times = n_eval_times
        self.best_ci_uno = -float('inf')
        self.save_dir = wandb.run.dir

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        model.eval()

        ci_unos, ibss, icis = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                m = self._evaluate_batch(batch, model)
                if m is None:
                    continue
                if not np.isnan(m['ci_uno']):
                    ci_unos.append(m['ci_uno'])
                if not np.isnan(m['ibs']):
                    ibss.append(m['ibs'])
                if not np.isnan(m['ici']):
                    icis.append(m['ici'])

        ci_uno = float(np.mean(ci_unos)) if ci_unos else float('nan')
        ibs    = float(np.mean(ibss))    if ibss    else float('nan')
        ici    = float(np.mean(icis))    if icis    else float('nan')

        if not np.isnan(ci_uno) and ci_uno > self.best_ci_uno:
            self.best_ci_uno = ci_uno
            self._save_checkpoint(model, epoch, loss, ci_uno,
                                  f"best_ciuno_epoch{epoch}_{ci_uno:.4f}.pt")
        if epoch % 50 == 0:
            self._save_checkpoint(model, epoch, loss, ci_uno,
                                  f"checkpoint_epoch{epoch}.pt")

        wandb.log({
            'epoch': epoch,
            'train/loss': loss,
            'train/epoch_time': epoch_time,
            'val/ci_uno': ci_uno,
            'val/ibs': ibs,
            'val/ici': ici,
            'val/best_ci_uno': self.best_ci_uno,
        }, step=epoch)
        print(
            f"epoch {epoch:5d} | {epoch_time:.1f}s | loss {loss:.3f} | "
            f"CI_uno {ci_uno:.3f} (best {self.best_ci_uno:.3f}) | "
            f"IBS {ibs:.4f} | ICI {ici:.4f}",
            flush=True,
        )

    def _evaluate_batch(self, batch, model):
        sep = batch['single_eval_pos']
        try:
            est, X_te, t_te, e_te, t_tr, e_tr = _build_est_from_batch(
                model, batch, sep, self.config, self.dist.borders, self.device,
            )
        except Exception:
            return None

        if not e_te.any() or len(t_te) < 3:
            return None

        eps = 1e-6
        t_lo = float(max(t_tr.min(), t_te.min())) + eps
        t_hi = float(min(t_tr.max(), t_te.max())) - eps
        if t_lo >= t_hi:
            return None
        eval_times = np.linspace(t_lo, t_hi, self.n_eval_times)

        surv_fns = est.predict_survival_function(X_te)
        surv_probs = np.stack([fn(eval_times) for fn in surv_fns])

        # SurvivalEVAL requires S(0)=1 as the first coordinate.
        eval_times_full = np.concatenate([[0.0], eval_times])
        surv_probs_full = np.concatenate(
            [np.ones((surv_probs.shape[0], 1)), surv_probs], axis=1,
        )

        risk_scores = est.predict(X_te)

        surv_tr = Surv.from_arrays(event=e_tr, time=t_tr)
        surv_te = Surv.from_arrays(event=e_te, time=t_te)

        try:
            ci_uno = float(concordance_index_ipcw(
                surv_tr, surv_te, risk_scores, tau=t_hi,
            )[0])
        except Exception:
            ci_uno = float('nan')

        ibs = ici = float('nan')
        try:
            evaluator = SurvivalEvaluator(
                pred_survs=surv_probs_full,
                time_coordinates=eval_times_full,
                event_times=t_te, event_indicators=e_te.astype(int),
                train_event_times=t_tr, train_event_indicators=e_tr.astype(int),
            )
        except Exception:
            return {'ci_uno': ci_uno, 'ibs': ibs, 'ici': ici}

        try:
            ibs = float(evaluator.integrated_brier_score(
                target_times=eval_times_full,
                IPCW_weighted=True, integration_method='trapz',
            ))
        except Exception:
            pass

        target_time = float(np.clip(
            np.median(np.concatenate([t_tr, t_te])), t_lo, t_hi,
        ))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary = evaluator.integrated_calibration_index(
                    target_time=target_time, knots=3, draw_figure=False,
                )
            ici = float(summary.get('ICI', float('nan')))
        except Exception:
            pass

        return {'ci_uno': ci_uno, 'ibs': ibs, 'ici': ici}

    def _save_checkpoint(self, model, epoch, loss, ci_uno, filename):
        path = f"{self.save_dir}/{filename}"
        if self.config is None:
            raise RuntimeError("ValidationLoggingCallback needs a SurvPFNConfig to save artifacts.")
        save_artifact(
            path,
            config=self.config,
            state_dict=model.state_dict(),
            bucket_edges=self.dist.borders,
            meta={'epoch': epoch, 'loss': float(loss), 'ci_uno': float(ci_uno)},
        )
        print(f"  Saved: {path}")

    def close(self):
        pass


class SurvSetEvalCallback(Callback):
    """Evaluate the model on real SurvSet datasets during training."""

    def __init__(self, dist: FullSupportBarDistribution, device,
                 config: SurvPFNConfig | None = None,
                 log_every: int = 10, survset_datasets=None,
                 min_rows=80, max_rows=1000, min_feat=2, max_feat=10,
                 n_folds=3, n_eval_times=50, seed=42):
        self.dist = dist
        self.device = device
        self.config = config
        self.log_every = log_every
        self.n_eval_times = n_eval_times
        self.seed = seed

        self.datasets = self._load_datasets(
            survset_datasets, min_rows, max_rows, min_feat, max_feat, n_folds, seed,
        )
        n_ds = len(set(d['dataset_name'] for d in self.datasets))
        print(f"[SurvSetEval] Loaded {len(self.datasets)} folds from {n_ds} datasets")

    @staticmethod
    def _load_datasets(survset_datasets, min_rows, max_rows, min_feat, max_feat, n_folds, seed):
        from .dataloader import (
            load_survset_dataset, list_survset_datasets, make_cv_splits,
        )

        if survset_datasets:
            ds_list = [(name, None, None) for name in survset_datasets]
        else:
            ds_list = list_survset_datasets(
                min_rows=min_rows, max_rows=max_rows, min_features=min_feat, max_features=max_feat,
                exclude_td=True,
            )

        datasets = []
        for ds_name, _, _ in ds_list:
            try:
                X_all, T_all, E_all, is_categorical = load_survset_dataset(ds_name)
            except Exception:
                continue

            if E_all.sum() < n_folds * 2:
                continue

            n = len(T_all)
            if max_rows and n > max_rows:
                rng = np.random.default_rng(seed)
                frac_ev = E_all.mean()
                n_ev = max(1, min(int(E_all.sum()), round(max_rows * frac_ev)))
                n_cens = max(1, max_rows - n_ev)
                ev_idx = np.where(E_all.astype(bool))[0]
                cens_idx = np.where(~E_all.astype(bool))[0]
                idx = np.concatenate([
                    rng.choice(ev_idx, min(n_ev, len(ev_idx)), replace=False),
                    rng.choice(cens_idx, min(n_cens, len(cens_idx)), replace=False),
                ])
                X_all, T_all, E_all = X_all[idx], T_all[idx], E_all[idx]

            splits = make_cv_splits(E_all, n_folds=n_folds, random_state=seed)
            for fold_idx, (tr_idx, te_idx) in enumerate(splits):
                datasets.append(dict(
                    X_train=X_all[tr_idx], X_test=X_all[te_idx],
                    t_train=T_all[tr_idx].astype(float),
                    t_test=T_all[te_idx].astype(float),
                    e_train=E_all[tr_idx].astype(bool),
                    e_test=E_all[te_idx].astype(bool),
                    is_categorical=np.asarray(is_categorical, dtype=bool),
                    dataset_name=ds_name, fold=fold_idx,
                    name=f'{ds_name}_fold{fold_idx}',
                ))

        return datasets

    def _run_tfm_on_ds(self, ds, model):
        t_tr, t_te = ds['t_train'], ds['t_test']
        e_tr, e_te = ds['e_train'], ds['e_test']
        X_tr, X_te = ds['X_train'], ds['X_test']

        if not e_te.any() or len(e_te) < 3:
            return None
        if e_tr.sum() < 3:
            return None

        eps = 1e-6
        t_lo = float(max(t_tr.min(), t_te.min())) + eps
        t_hi = float(min(t_tr.max(), t_te.max())) - eps
        if t_lo >= t_hi:
            return None
        eval_times = np.linspace(t_lo, t_hi, self.n_eval_times)

        cat_indices = np.where(ds['is_categorical'])[0].tolist() or None
        est = SurvPFN.from_live_model(
            model=model, config=self.config, bucket_edges=self.dist.borders,
            device=self.device, categorical_features_indices=cat_indices,
        )
        y_tr = Surv.from_arrays(event=e_tr, time=t_tr)
        est.fit(X_tr, y_tr)

        surv_fns = est.predict_survival_function(X_te)
        surv_probs = np.stack([fn(eval_times) for fn in surv_fns])
        risk_scores = est.predict(X_te)

        try:
            ci = float(concordance_index_censored(e_te, t_te, risk_scores)[0])
        except Exception:
            ci = float('nan')

        ibs = float('nan')
        surv_tr_struct = Surv.from_arrays(event=e_tr, time=t_tr)
        surv_te_struct = Surv.from_arrays(event=e_te, time=t_te)
        try:
            ibs = float(integrated_brier_score(
                surv_tr_struct, surv_te_struct, surv_probs, eval_times,
            ))
        except Exception:
            pass

        nll_events = float('nan')
        try:
            ev_mask = torch.from_numpy(e_te.astype(bool)).to(self.device)
            if ev_mask.any():
                logits = est.predict_logits(X_te)
                t_te_t = torch.from_numpy(t_te.astype(np.float32)).to(self.device)
                y_ev = est.transform_times(t_te_t[ev_mask])
                nll_per = self.dist(logits[ev_mask].unsqueeze(0), y_ev.unsqueeze(0))
                nll_events = float(nll_per.mean().item())
        except Exception:
            pass

        return {'ci': ci, 'ibs': ibs, 'nll_events': nll_events}

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        if epoch % self.log_every != 0:
            return

        model.eval()

        per_ds = {}
        all_ci, all_ibs, all_nll = [], [], []

        with torch.no_grad():
            for ds in self.datasets:
                try:
                    result = self._run_tfm_on_ds(ds, model)
                except Exception:
                    continue
                if result is None:
                    continue
                ds_name = ds['dataset_name']
                if ds_name not in per_ds:
                    per_ds[ds_name] = {'ci': [], 'ibs': [], 'nll_events': []}
                for key, pool in (('ci', all_ci), ('ibs', all_ibs), ('nll_events', all_nll)):
                    val = result[key]
                    if not np.isnan(val):
                        per_ds[ds_name][key].append(val)
                        pool.append(val)

        ds_ci_means  = [np.mean(v['ci'])         for v in per_ds.values() if v['ci']]
        ds_ibs_means = [np.mean(v['ibs'])        for v in per_ds.values() if v['ibs']]
        ds_nll_means = [np.mean(v['nll_events']) for v in per_ds.values() if v['nll_events']]

        def _agg(values):
            if not values:
                return float('nan'), float('nan'), float('nan')
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(np.median(arr)), float(arr.std(ddof=0))

        macro_ci,  med_ci,  std_ci  = _agg(ds_ci_means)
        macro_ibs, med_ibs, std_ibs = _agg(ds_ibs_means)
        macro_nll, med_nll, std_nll = _agg(ds_nll_means)

        micro_ci  = float(np.mean(all_ci))  if all_ci  else float('nan')
        micro_ibs = float(np.mean(all_ibs)) if all_ibs else float('nan')
        micro_nll = float(np.mean(all_nll)) if all_nll else float('nan')

        wandb.log({
            'epoch': epoch,
            'survset/c_index':       micro_ci,
            'survset/ibs':           micro_ibs,
            'survset/n_datasets':    len(per_ds),

            'survset/agg/c_index_macro':    macro_ci,
            'survset/agg/c_index_median':   med_ci,
            'survset/agg/c_index_std':      std_ci,
            'survset/agg/c_index_micro':    micro_ci,

            'survset/agg/ibs_macro':        macro_ibs,
            'survset/agg/ibs_median':       med_ibs,
            'survset/agg/ibs_std':          std_ibs,
            'survset/agg/ibs_micro':        micro_ibs,

            'survset/agg/nll_events_macro':  macro_nll,
            'survset/agg/nll_events_median': med_nll,
            'survset/agg/nll_events_std':    std_nll,
            'survset/agg/nll_events_micro':  micro_nll,

            'survset/agg/n_datasets':       len(per_ds),
            'survset/agg/n_folds':          len(all_ci),
        }, step=epoch)

        table_data = []
        per_ds_log = {'epoch': epoch}
        for ds_name, vals in sorted(per_ds.items()):
            ds_ci  = float(np.mean(vals['ci']))         if vals['ci']         else float('nan')
            ds_ibs = float(np.mean(vals['ibs']))        if vals['ibs']        else float('nan')
            ds_nll = float(np.mean(vals['nll_events'])) if vals['nll_events'] else float('nan')
            table_data.append([ds_name, ds_ci, ds_ibs, ds_nll, len(vals['ci'])])
            per_ds_log[f'survset/{ds_name}/c_index']    = ds_ci
            per_ds_log[f'survset/{ds_name}/ibs']        = ds_ibs
            per_ds_log[f'survset/{ds_name}/nll_events'] = ds_nll
        wandb.log(per_ds_log, step=epoch)

        table = wandb.Table(
            columns=['dataset', 'c_index', 'ibs', 'nll_events', 'n_folds'],
            data=table_data,
        )
        wandb.log({'survset/per_dataset': table, 'epoch': epoch}, step=epoch)

        print(
            f"  [SurvSet] epoch {epoch} | "
            f"C-index macro={macro_ci:.3f} (med={med_ci:.3f}, std={std_ci:.3f}) | "
            f"IBS macro={macro_ibs:.4f} (med={med_ibs:.4f}) | "
            f"NLL_ev macro={macro_nll:.3f} | "
            f"{len(per_ds)} datasets, {len(all_ci)} folds",
            flush=True,
        )

    def close(self):
        pass
