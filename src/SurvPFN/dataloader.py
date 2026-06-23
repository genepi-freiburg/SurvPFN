import h5py
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
    
## Dataloader for Prior

class PriorMultifileDataloader(DataLoader):
    """Training loader for per-ncols HDF5 files.

    Every batch has a constant shape — same n_rows and n_cols for all datasets
    in the batch, no padding. Between batches, both are re-sampled independently.

    Parameters
    ----------
    file_pattern : str
        Format string with a ``{ncols}`` placeholder, e.g.
        ``'/data/prior_ncols_{ncols:02d}.h5'``.
    ncols_range : (int, int)
        Min/max number of features to sample per batch (inclusive).
    nrows_range : (int, int)
        Min/max number of rows (samples) to sample per batch (inclusive).
    train_frac_range : (float, float)
        Min/max fraction of rows used as training context.  The separator
        ``single_eval_pos = int(n_rows * train_frac)`` is constant within a batch.
    batch_size : int
    num_steps : int
    device : str or torch.device
    track_coverage : bool, optional
        If True, tracks which datasets have been seen and logs coverage statistics.
        Defaults to False.
    """

    def __init__(
        self,
        file_pattern: str,
        ncols_range: tuple,
        nrows_range: tuple,
        train_frac_range: tuple,
        batch_size: int,
        num_steps: int,
        device,
        track_coverage=False,
    ):
        self.file_pattern     = file_pattern
        self.ncols_range      = ncols_range
        self.nrows_range      = nrows_range
        self.train_frac_range = train_frac_range
        self.batch_size       = batch_size
        self.num_steps        = num_steps
        self.device           = device
        self.track_coverage   = track_coverage

        # Cache dataset counts so we don't re-open files on every step
        self._n_datasets: dict[int, int] = {}
        for ncols in range(ncols_range[0], ncols_range[1] + 1):
            path = file_pattern.format(ncols=ncols)
            with h5py.File(path, 'r') as f:
                self._n_datasets[ncols] = int(f['n_datasets'][()])

        # Initialize coverage tracking
        if self.track_coverage:
            self._seen_datasets: dict[int, set] = {ncols: set() for ncols in self._n_datasets.keys()}
            self._total_datasets = sum(self._n_datasets.values())

    def __iter__(self):
        for step in range(self.num_steps):
            ncols      = int(np.random.randint(self.ncols_range[0], self.ncols_range[1] + 1))
            n_rows     = int(np.random.randint(self.nrows_range[0], self.nrows_range[1] + 1))
            train_frac = float(np.random.uniform(*self.train_frac_range))
            sep        = int(n_rows * train_frac)

            # Sample dataset indices
            n_total = self._n_datasets[ncols]
            indices = np.sort(np.random.choice(n_total, size=self.batch_size, replace=False))

            # Track coverage if enabled
            if self.track_coverage:
                self._seen_datasets[ncols].update(indices)

            path = self.file_pattern.format(ncols=ncols)
            with h5py.File(path, 'r') as f:
                x                = torch.from_numpy(np.array(f['X'][indices,                :n_rows, :]))
                observed_time    = torch.from_numpy(np.array(f['observed_time'][indices,    :n_rows]))
                true_event_time  = torch.from_numpy(np.array(f['true_event_time'][indices,  :n_rows]))
                event_indicator  = torch.from_numpy(np.array(f['event_indicator'][indices,  :n_rows]))
                linear_predictor = torch.from_numpy(np.array(f['linear_predictor'][indices, :n_rows]))
                
                wk_per_ds = torch.from_numpy(np.array(f["meta/weibull_k"][indices])).float()
                wl_per_ds = torch.from_numpy(np.array(f["meta/weibull_lambda"][indices])).float()
                wk = float(wk_per_ds[0])
                wl = float(wl_per_ds[0])
                
                # Check if the dataset is non-proportional hazards (NPH) based on metadata
                is_nph = bool(int(f["meta/is_nph"][indices[0]]))
                if 'per_patient_k' in f:
                    per_patient_k = torch.from_numpy(np.array(f['per_patient_k'][indices, :n_rows]))
                else:
                    per_patient_k = torch.full((len(indices), n_rows), float('nan'))
                
                is_categorical = torch.from_numpy(np.array(f["is_categorical"][indices]))
                is_categorical = is_categorical.bool()
                
            # Log coverage every 100 steps if tracking
            if self.track_coverage and step % 100 == 0:
                total_seen = sum(len(s) for s in self._seen_datasets.values())
                coverage_pct = (total_seen / self._total_datasets) * 100 if self._total_datasets > 0 else 0
                print(f"Step {step}/{self.num_steps}: Coverage = {coverage_pct:.2f}% ({total_seen}/{self._total_datasets} datasets)")

            yield dict(
                x=x.to(self.device),
                observed_time=observed_time.to(self.device),
                true_event_time=true_event_time.to(self.device),
                event_indicator=event_indicator.to(self.device),
                linear_predictor=linear_predictor.to(self.device),
                per_patient_k=per_patient_k.to(self.device),
                single_eval_pos=sep,
                weibull_k=wk,
                weibull_lambda=wl,
                weibull_k_per_ds=wk_per_ds.to(self.device),
                weibull_lambda_per_ds=wl_per_ds.to(self.device),
                is_nph=is_nph,
                is_categorical=is_categorical.to(self.device),
            )

    def get_coverage_stats(self):
        """Return coverage statistics.
        
        Returns:
            dict: Dictionary with coverage information including:
                - total_datasets: Total number of datasets across all ncols
                - total_seen: Total number of unique datasets seen
                - coverage_percentage: Percentage of datasets covered
                - coverage_by_ncols: Detailed coverage per ncols value
        """
        if not self.track_coverage:
            return {"error": "Coverage tracking not enabled"}
        
        total_seen = sum(len(s) for s in self._seen_datasets.values())
        coverage_pct = (total_seen / self._total_datasets) * 100 if self._total_datasets > 0 else 0
        
        coverage_by_ncols = {}
        for ncols, seen_set in self._seen_datasets.items():
            coverage_by_ncols[ncols] = {
                "total": self._n_datasets[ncols],
                "seen": len(seen_set),
                "percentage": (len(seen_set) / self._n_datasets[ncols] * 100) if self._n_datasets[ncols] > 0 else 0
            }
        
        return {
            "total_datasets": self._total_datasets,
            "total_seen": total_seen,
            "coverage_percentage": coverage_pct,
            "coverage_by_ncols": coverage_by_ncols
        }

    def __len__(self):
        return self.num_steps
    
## Loader functions for SurvSet

def make_cv_splits(E, n_folds=5, random_state=42):
    """Stratified CV splits on event indicator."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(E)), E.astype(int)))


def list_survset_datasets(min_rows=None, max_rows=None, min_features=None, max_features=None,
                          exclude_td=True):
    """Return a list of (ds_name, n, n_features) for qualifying SurvSet datasets.

    Parameters
    ----------
    exclude_td            : drop time-varying datasets (default True)
    """
    from SurvSet.data import SurvLoader
    loader = SurvLoader()
    df = loader.df_ds.copy()
    if exclude_td:
        df = df[~df['is_td']]

    df['n_features'] = df['n_num'] + df['n_fac']
    if min_rows is not None:
        df = df[df['n'] >= min_rows]
    if max_rows is not None:
        df = df[df['n'] <= max_rows]
    
    if min_features is not None:
        df = df[df['n_features'] >= min_features]
    if max_features:
        df = df[df['n_features'] <= max_features]
    return list(zip(df['ds'].tolist(), df['n'].tolist(), df['n_features'].tolist()))


def load_survset_dataset(ds_name):
    """
    Load a SurvSet dataset and return features, labels, event indicator and categorical mask.
    
    Returns
    -------
    X              : (n, p) float32 features
    T              : (n,) float32 event times
    E              : (n,) float32 event indicators
    is_categorical : (p,) boolean mask (True if column is categorical/OHE)
    """
    from SurvSet.data import SurvLoader
    data = SurvLoader().load_dataset(ds_name=ds_name)
    df = data['df'].drop(columns=['pid', 'time2'], errors='ignore')

    cat_cols = [c for c in df.columns if c.startswith('fac_')]
    num_cols = [c for c in df.columns if c.startswith('num_')]
    
    X_df = df[num_cols].copy()
    for col in cat_cols:
        codes = df[col].astype('category').cat.codes.to_numpy()
        n_cat = int(codes.max()) + 1 if (codes >= 0).any() else 0
        codes = np.where(codes == -1, n_cat, codes).astype(np.float32)
        X_df[col] = codes
    is_categorical = [False] * len(num_cols) + [True] * len(cat_cols)
    #print(f"N categorical features (encoded as numeric): {len(cat_cols)}")
    #print(f"N cathegories per categorical feature: {[df[c].nunique(dropna=True) for c in cat_cols]}")

    X = X_df.values.astype(np.float32)
    T = df['time'].values.astype(np.float32)
    E = df['event'].values.astype(np.float32)
    is_categorical = np.array(is_categorical, dtype=bool)

    # Median-impute NaN in continuous columns only.
    for j in range(X.shape[1]):
        if is_categorical[j]:
            continue
        mask = np.isnan(X[:, j])
        if mask.any():
            fill_val = np.nanmedian(X[:, j])
            X[mask, j] = float(fill_val)
    

    return X, T, E, is_categorical