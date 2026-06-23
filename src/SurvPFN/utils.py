import h5py
import numpy as np
import torch
from pfns.bar_distribution import get_bucket_limits

from .transforms import log1p_transform


def _enforce_min_bucket_width(be: torch.Tensor, min_width: float = 1e-6) -> torch.Tensor:
    """Nudge adjacent edges apart so every bucket has width >= min_width.

    ``get_bucket_limits`` pins the outer edges to ``ys.min()/ys.max()``; when
    those values are duplicated in the pool, the first/last quantile edge can
    coincide with the outer edge → width 0 → HalfNormal(scale=0) crashes.
    """
    be_np = be.detach().cpu().numpy().copy()
    for j in range(1, len(be_np)):
        if be_np[j] - be_np[j - 1] < min_width:
            be_np[j] = be_np[j - 1] + min_width
    return torch.tensor(be_np, device=be.device, dtype=be.dtype)

def compute_bucket_edges_multifile(
    file_pattern, ncols_range, n_buckets=1000, device='cpu',
    max_datasets_per_ncols=25000, y_field='true_event_time',
):
    """Quantile bucket edges from per-ncols HDF5 files.

    Samples ``max_datasets_per_ncols`` datasets equally from each ncols file,
    applies per-dataset z-normalization (same logic as training), and computes
    quantile bucket edges over the pooled normalized values.

    The per-ncols files have shape ``(n_datasets, max_rows, ncols)`` with no
    ``single_eval_pos`` / ``num_datapoints`` metadata — those are sampled here
    from ``nrows_range`` / ``train_frac_range`` to match the training distribution.
    """

    all_normalized = []

    for ncols in range(ncols_range[0], ncols_range[1] + 1):
        path = file_pattern.format(ncols=ncols)
        with h5py.File(path, 'r') as f:
            n_total = int(f['n_datasets'][()])
            n_use = min(n_total, max_datasets_per_ncols)
            indices = np.sort(np.random.choice(n_total, size=n_use, replace=False))

            y_raw = torch.tensor(np.array(f[y_field][indices]), dtype=torch.float32)
            
            ev_ind = torch.tensor(np.array(f['event_indicator'][indices]), dtype=torch.bool)

        for i in range(n_use):
            all_targets = log1p_transform(y_raw[i])

            mask = ev_ind[i]
            if mask.sum() < 10:
                continue
            ref = all_targets[mask]
        

            ref = ref[torch.isfinite(ref)]
            if len(ref) < 10:
                continue

            mean = ref.mean()
            std = ref.std() + 1e-8
            all_normalized.append((ref - mean) / std)

    ys = torch.cat(all_normalized)
    ys = ys[torch.isfinite(ys)]

    be = get_bucket_limits(n_buckets, ys=ys).to(device)
    return _enforce_min_bucket_width(be)