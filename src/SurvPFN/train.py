"""SurvPFN training entry point.

Mode is composed from three orthogonal flags:

  --loss {oracle,native}    base loss formulation
                            oracle: NLL on log1p(true_event_time) at test rows
                            native: IPCW-weighted NLL on log1p(observed_time)
                                    + right-censored NLL = -log S(t_c)
  --rank                    add DeepHit-style pairwise ranking loss
  --ablation                feed event-indicator feature column = 1.0 (model
                            blind to censoring)

The loss object (see ``losses.py``) declares which extras the train loop
must compute via ``needs_ipcw`` / ``needs_linear_predictor`` flags.
"""

import argparse
import os
import time

import torch
import schedulefree
import wandb

from pfns.bar_distribution import FullSupportBarDistribution
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device, set_randomness_seed

from .dataloader import PriorMultifileDataloader

from .model import SurvPFNNet
from .utils import compute_bucket_edges_multifile
from .loss import km_censoring_weights, SurvivalLoss, build_loss
from .callbacks import ValidationLoggingCallback, SurvSetEvalCallback
from .transforms import events_only_stats
from .config import SurvPFNConfig
from ._artifact import save_artifact, load_artifact


def _make_prepare_batch(loss_obj: SurvivalLoss, ablation: bool, device):
    """Return ``prepare_batch(raw_batch) -> (x, y, ev_test)``.

    Builds the model input ``x`` (raw features + appended event-indicator
    column), delegates y-target construction to ``loss_obj.y_target``, and
    extracts the real test-row event indicator ``ev_test`` (used as
    test_mask in the loss and in ranking).

    Under ``ablation`` the appended event-indicator column is forced to 1.0
    so the model is blind to censoring; the loss/test_mask still see the
    real censoring pattern.
    """

    def prepare(batch):
        x_raw  = batch['x'].to(device)
        obs    = batch['observed_time'].to(device)
        ev     = batch['event_indicator'].to(device)
        true_evt = batch.get('true_event_time')
        if true_evt is not None:
            true_evt = true_evt.to(device)
        is_cat = batch.get('is_categorical')
        sep = batch['single_eval_pos']
        B, n, F = x_raw.shape

        ev_col = torch.ones(B, n, 1, device=device)
        if not ablation:
            ev_col[:, :sep, 0] = ev[:, :sep]
        x = torch.cat([x_raw, ev_col], dim=-1)

        if is_cat is not None:
            cat_mask = torch.cat([
                is_cat.to(device).bool().reshape(-1, F)[0],
                torch.ones(1, dtype=torch.bool, device=device),
            ])
        else:
            cat_mask = torch.zeros(F + 1, dtype=torch.bool, device=device)
            cat_mask[-1] = True
        batch['_cat_mask_ext'] = cat_mask

        y = loss_obj.y_target(obs, true_evt, sep)
        ev_test = ev[:, sep:].bool()
        return x, y, ev_test

    return prepare


def train(
    model,
    prior,
    criterion: FullSupportBarDistribution,
    prepare_batch,
    loss_obj: SurvivalLoss,
    epochs: int,
    lr: float,
    device,
    config: SurvPFNConfig,
    bucket_edges: torch.Tensor,
    callbacks=None,
    ckpt=None,
    accumulate_gradients: int = 1,
    multi_gpu: bool = False,
):
    if callbacks is None:
        callbacks = []
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)

    assert prior.num_steps % accumulate_gradients == 0, \
        'num_steps must be divisible by accumulate_gradients'

    start_epoch = ckpt['epoch'] + 1 if ckpt else 1
    total_loss = 0.0
    n_batches = 0

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            model.train()
            optimizer.train()
            total_loss = 0.0
            total_grad_norm = 0.0
            total_loss_info = {}
            n_batches = 0
            event_skipped, std_skipped, nan_skipped = 0, 0, 0
            n_optimizer_steps = 0

            for raw_batch in prior:
                x, y, ev_test = prepare_batch(raw_batch)
                sep = raw_batch['single_eval_pos']

                ev_train = raw_batch['event_indicator'].to(x.device)[:, :sep]

                # Skip datasets with too few events for stable z-normalization.
                if (ev_train.sum(dim=1) < 3).any():
                    event_skipped += 1
                    continue

                mean, std = events_only_stats(y[:, :sep], ev_train)
                y_norm = (y - mean) / std

                # Guard against degenerate per-dataset stats.
                if (std < 1e-3).any():
                    std_skipped += 1
                    continue


                y_norm = y_norm.clamp(-20.0, 20.0)
                y_target = y_norm[:, sep:]

                if (torch.isnan(x).any()
                        or torch.isnan(y_norm[:, :sep]).any()
                        or torch.isnan(y_target).any()):
                    nan_skipped += 1
                    continue

                ipcw_w = None
                if loss_obj.needs_ipcw:
                    ipcw_w = km_censoring_weights(
                        raw_batch['observed_time'].to(x.device),
                        raw_batch['event_indicator'].to(x.device),
                        sep,
                    )

                lp_test = None
                if loss_obj.needs_linear_predictor:
                    lp_test = raw_batch['linear_predictor'].to(x.device)[:, sep:]

                output = model(
                    (x, y_norm[:, :sep]),
                    single_eval_pos=sep,
                    cat_mask=raw_batch.get('_cat_mask_ext'),
                )

                loss_val, loss_info = loss_obj.compute(
                    output, y_target, ev_test,
                    ipcw=ipcw_w, linear_predictor=lp_test,
                )

                loss_val = loss_val / accumulate_gradients
                loss_val.backward()
                total_loss += loss_val.item() * accumulate_gradients
                for k, v in loss_info.items():
                    total_loss_info[k] = total_loss_info.get(k, 0.0) + v
                n_batches += 1

                if n_batches % accumulate_gradients == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    total_grad_norm += grad_norm.item()
                    n_optimizer_steps += 1
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_time = time.time() - epoch_start

            if n_batches == 0:
                print(f"  epoch {epoch}: all batches skipped!")
                continue

            mean_loss = total_loss / n_batches
            mean_grad_norm = total_grad_norm / max(n_optimizer_steps, 1)
            model.eval()
            optimizer.eval()

            log_dict = {
                'epoch': epoch,
                'train/loss': mean_loss,
                'train/epoch_time': epoch_time,
                'train/grad_norm': mean_grad_norm,
                'train/event_skipped': event_skipped,
                'train/std_skipped': std_skipped,
                'train/nan_skipped': nan_skipped,
            }
            for k, v in total_loss_info.items():
                log_dict[f'train/{k}'] = v / n_batches
            wandb.log(log_dict, step=epoch)

            _m = model.module if multi_gpu else model
            save_artifact(
                f"{wandb.run.dir}/latest.pt",
                config=config,
                state_dict=_m.state_dict(),
                bucket_edges=bucket_edges,
                meta={'epoch': epoch},
            )

            for callback in callbacks:
                callback.on_epoch_end(epoch, epoch_time, mean_loss, _m, dist=criterion)

            if epoch <= 3 or epoch % 10 == 0:
                print(
                    f'  epoch {epoch:5d} | time {epoch_time:.2f}s | '
                    f'loss {mean_loss:.4f} | grad_norm {mean_grad_norm:.4f} | '
                    f'batches {n_batches} | skipped {event_skipped}/{std_skipped}/{nan_skipped}',
                    flush=True,
                )

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        for callback in callbacks:
            callback.close()

    _m = model.module if multi_gpu else model
    return _m, total_loss / max(n_batches, 1)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train SurvPFN with on survival priors."
    )

    # Mode (orthogonal flags).
    parser.add_argument("--loss", required=True, choices=['oracle', 'native'],
                        help="oracle: NLL on oracle true_event_time | "
                             "native: IPCW NLL + censored NLL on observed_time")
    parser.add_argument("--rank", action="store_true",
                        help="Add DeepHit-style pairwise ranking loss")
    parser.add_argument("--ablation", action="store_true",
                        help="Feed event-indicator feature column = 1.0 "
                             "(model blind to censoring). Disables dual encoder "
                             "for --loss native.")
    parser.add_argument("--rank_weight", type=float, default=1.0,
                        help="Weight on the ranking loss term")

    # Data.
    parser.add_argument("--file_pattern", type=str, required=True)
    parser.add_argument("--val_pattern", type=str, required=True)
    parser.add_argument("--ncols_min", type=int, default=2)
    parser.add_argument("--ncols_max", type=int, default=10)
    parser.add_argument("--nrows_min", type=int, default=100)
    parser.add_argument("--nrows_max", type=int, default=1000)
    parser.add_argument("--train_frac_min", type=float, default=0.5)
    parser.add_argument("--train_frac_max", type=float, default=0.9)

    # Architecture.
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--embeddingsize", type=int, default=192)
    parser.add_argument("--hiddensize", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--n_buckets", type=int, default=1000)

    # Training.
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--loadcheckpoint", type=str, default=None)

    # Logging.
    parser.add_argument("--wandb_project", type=str, default="survival-tabpfn")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def _setup_data(args, device):
    ncols_range = (args.ncols_min, args.ncols_max)
    nrows_range = (args.nrows_min, args.nrows_max)
    train_frac_range = (args.train_frac_min, args.train_frac_max)

    prior = PriorMultifileDataloader(
        file_pattern=args.file_pattern,
        ncols_range=ncols_range, nrows_range=nrows_range,
        train_frac_range=train_frac_range,
        batch_size=args.batchsize, num_steps=args.steps, device=device,
    )
    val_loader = PriorMultifileDataloader(
        file_pattern=args.val_pattern,
        ncols_range=ncols_range, nrows_range=nrows_range,
        train_frac_range=train_frac_range,
        batch_size=1, num_steps=250, device=device,
    )
    bucket_edges = compute_bucket_edges_multifile(
        file_pattern=args.file_pattern,
        ncols_range=ncols_range,
        n_buckets=args.n_buckets, device=device,
        y_field='observed_time',
    )
    print(f"Bucket edges: {args.n_buckets} quantile buckets | log1p / events_only")

    dist = FullSupportBarDistribution(bucket_edges)

    return prior, val_loader, dist


def _setup_model(args, n_primary, ckpt):
    base_model = SurvPFNNet(
        num_attention_heads=args.heads,
        embedding_size=args.embeddingsize,
        mlp_hidden_size=args.hiddensize,
        num_layers=args.layers,
        num_outputs=n_primary,
    )
    if ckpt:
        base_model.load_state_dict(ckpt['model'], strict=False)
    return base_model


def _setup_callbacks(val_loader, dist, device, config):
    return [
        ValidationLoggingCallback(
            val_loader=val_loader, device=device, dist=dist, config=config,
        ),
        SurvSetEvalCallback(
            dist=dist, device=device, config=config,
            log_every=25, n_folds=3, min_rows=1, max_rows=1000, min_feat=2, max_feat=10,
            seed=2026,
        ),
    ]


def main():
    args = _parse_args()

    # CUDA allocator: variable batch shapes (nrows/ncols vary per step) cause
    # fragmentation otherwise. Must be set before any CUDA initialisation.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    set_randomness_seed(2026)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               config=vars(args))

    print(f"SurvPFN training | loss={args.loss} rank={args.rank} ablation={args.ablation}")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    device = get_default_device()

    ckpt = None
    if args.loadcheckpoint:
        _, state_dict_loaded, _, meta_loaded = load_artifact(args.loadcheckpoint)
        ckpt = {
            'epoch': int(meta_loaded.get('epoch', 0)),
            'model': state_dict_loaded,
        }

    prior, val_loader, dist = _setup_data(args, device)

    cfg = SurvPFNConfig(
        embedding_dim=args.embeddingsize,
        hidden_dim=args.hiddensize,
        n_attention_heads=args.heads,
        n_layers=args.layers,
        n_buckets=args.n_buckets,
        loss_type=args.loss,
        use_ranking=args.rank,
        ranking_weight=args.rank_weight,
        is_ablation=args.ablation,
    )

    loss_obj = build_loss(
        args.loss, criterion=dist,
        rank=args.rank, rank_weight=args.rank_weight,
    )

    prepare_batch = _make_prepare_batch(loss_obj, args.ablation, device)
    model = _setup_model(args, args.n_buckets, ckpt)

    callbacks = _setup_callbacks(val_loader, dist, device, cfg)

    trained_model, _ = train(
        model=model, prior=prior, criterion=dist,
        prepare_batch=prepare_batch, loss_obj=loss_obj,
        epochs=args.epochs, lr=args.lr,
        device=device, config=cfg, bucket_edges=dist.borders,
        callbacks=callbacks, ckpt=ckpt,
        accumulate_gradients=args.accumulate,
    )

    save_path = f"{wandb.run.dir}/final.pt"
    save_artifact(
        save_path,
        config=cfg,
        state_dict=trained_model.state_dict(),
        bucket_edges=dist.borders,
        meta={'epoch': args.epochs},
    )
    print(f"Saved final model to {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
