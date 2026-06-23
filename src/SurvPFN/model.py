"""SurvPFN model: censoring-aware survival extension of NanoTabPFN.

Builds on ``tfmplayground.model.NanoTabPFNModel`` (pinned in ``pyproject.toml``)
"""

from typing import Optional, Tuple

import torch
from torch import nn
from tfmplayground.model import FeatureEncoder, NanoTabPFNModel, TargetEncoder


class CategoricalAwareFeatureEncoder(FeatureEncoder):
    """Feature encoder with a separate branch for categorical columns.

    Continuous columns reuse the upstream behaviour (per-batch z-norm w.r.t.
    train rows, clip to ±100, shared ``nn.Linear(1, E)`` named ``linear_layer``).

    Categorical columns go through a fresh ``nn.Linear(1, E)`` named
    ``cat_linear`` applied to the raw integer codes.

    The split is driven by ``cat_mask`` passed to ``forward`` — a boolean
    tensor of shape ``(num_features,)`` (or ``(B, num_features)``; the first
    row is used). ``cat_mask=None`` falls back to the upstream encoder
    verbatim, so checkpoints saved before the categorical branch existed load
    cleanly with ``strict=False``.
    """

    def __init__(self, embedding_size: int):
        super().__init__(embedding_size)
        self.cat_linear = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int,
                cat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cat_mask is None or not cat_mask.any():
            return super().forward(x, single_eval_pos)

        if cat_mask.dim() == 2:
            cat_mask = cat_mask[0]
        cat_mask = cat_mask.bool().to(x.device)
        cont_mask = ~cat_mask

        E = self.linear_layer.out_features
        out = x.new_zeros(x.shape[0], x.shape[1], x.shape[-1], E)

        if cont_mask.any():
            out[:, :, cont_mask, :] = super().forward(x[:, :, cont_mask], single_eval_pos)

        x_cat = x[:, :, cat_mask].float().unsqueeze(-1)
        out[:, :, cat_mask, :] = self.cat_linear(x_cat)
        return out


class SurvPFNNet(NanoTabPFNModel):
    """NanoTabPFN with a second target encoder for censored training rows.

    Train rows are embedded by blending the event and censored target encoders
    using the event indicator, so the model sees different representations for
    events vs. censored observations.
    """

    def __init__(self, embedding_size: int, num_attention_heads: int,
                 mlp_hidden_size: int, num_layers: int, num_outputs: int):
        super().__init__(embedding_size, num_attention_heads, mlp_hidden_size,
                         num_layers, num_outputs)
        self.feature_encoder = CategoricalAwareFeatureEncoder(embedding_size)
        self.target_encoder_censored = TargetEncoder(embedding_size)

    def forward(self, *args, cat_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if len(args) == 3:
            x_train = args[0]
            x = x_train if args[2] is None else torch.cat((x_train, args[2]), dim=1)
            return self._forward((x, args[1]),
                                 single_eval_pos=x_train.shape[1],
                                 cat_mask=cat_mask, **kwargs)
        if len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(args[0], cat_mask=cat_mask, **kwargs)
        raise ValueError("Unsupported input format")

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor],
                 single_eval_pos: int, num_mem_chunks: int = 1,
                 cat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.get_embeddings(src, single_eval_pos,
                                         num_mem_chunks=num_mem_chunks,
                                         cat_mask=cat_mask)
        return self.decoder(embeddings)

    def get_embeddings(self, src: Tuple[torch.Tensor, torch.Tensor],
                       single_eval_pos: int, num_mem_chunks: int = 1,
                       cat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the backbone up to (but not including) the decoder.

        Returns ``(B, n_test, E)`` embeddings for test positions.
        """
        x_src, y_src = src
        if y_src.dim() < x_src.dim():
            y_src = y_src.unsqueeze(-1)

        B, num_rows = x_src.shape[0], x_src.shape[1]
        ev_ind = x_src[:, :single_eval_pos, -1]

        if cat_mask is not None:
            x_emb = self.feature_encoder(x_src, single_eval_pos, cat_mask=cat_mask)
        else:
            x_emb = self.feature_encoder(x_src, single_eval_pos)

        y_emb_event = self.target_encoder(y_src, num_rows)
        y_emb_cens = self.target_encoder_censored(y_src, num_rows)

        blend = torch.ones(B, num_rows, 1, 1, device=x_src.device)
        blend[:, :single_eval_pos, 0, 0] = ev_ind
        y_emb = blend * y_emb_event + (1 - blend) * y_emb_cens

        src_cat = torch.cat([x_emb, y_emb], dim=2)
        output = self.transformer_encoder(src_cat, single_eval_pos,
                                          num_mem_chunks=num_mem_chunks)
        return output[:, single_eval_pos:, -1, :]
