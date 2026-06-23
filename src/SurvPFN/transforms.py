"""Target preprocessing: log1p + events-only z-normalisation."""

import torch


def log1p_transform(time: torch.Tensor) -> torch.Tensor:
    return torch.log1p(time.clamp(min=0))


def expm1_inverse(t: torch.Tensor) -> torch.Tensor:
    return torch.expm1(t)


def events_only_stats(y_log: torch.Tensor, event: torch.Tensor):
    """Mean/std of y_log restricted to uncensored events.

    Reduces over the last dim. Works for (n,) and (B, n).
    """
    mask = event.bool().float()
    n = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = (y_log * mask).sum(dim=-1, keepdim=True) / n
    var = ((y_log - mean) * mask).pow(2).sum(dim=-1, keepdim=True) / (n - 1).clamp(min=1)
    std = var.sqrt().clamp(min=1e-6)
    return mean, std


class TargetTransform:
    """Stateful log1p + events-only z-norm for the SurvPFN estimator."""

    def fit(self, time: torch.Tensor, event: torch.Tensor) -> "TargetTransform":
        time = torch.as_tensor(time, dtype=torch.float32)
        event = torch.as_tensor(event, dtype=torch.float32)
        if event.sum() < 3:
            raise ValueError("Need at least 3 uncensored events to fit.")
        y_log = log1p_transform(time)
        mean, std = events_only_stats(y_log, event)
        self.mean_ = mean.squeeze()
        self.std_ = std.squeeze()
        return self

    def transform(self, time: torch.Tensor) -> torch.Tensor:
        time = torch.as_tensor(time, dtype=torch.float32, device=self.mean_.device)
        return (log1p_transform(time) - self.mean_) / self.std_

    def inverse(self, t_norm: torch.Tensor) -> torch.Tensor:
        return expm1_inverse(t_norm * self.std_ + self.mean_)
