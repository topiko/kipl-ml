"""Causal, finite-horizon TAM classifier with the discriminator's per-bin API."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from kipl_ml.defences.models.trgen import (
    _feature_map,
    select_tam_prediction,
    tam_seq_len_fun,
)
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.tam_stats import TAM_TIME_STD_FEATURES


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent query/key pairs; x is (batch, heads, tokens, head_dim)."""
    head_dim = x.shape[-1]
    frequencies = 10000.0 ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
    )
    angles = positions.float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class RoPEEncoderLayer(nn.Module):
    """Pre-norm encoder with rotary Q/K and explicit causal/padding masks."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # Match MultiheadAttention's projection initialization for the comparison.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        self.out_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_projection.bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, src_key_padding_mask):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(
            batch, tokens, 3, self.nhead, self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        positions = torch.arange(tokens, device=x.device)
        query, key = apply_rope(query, positions), apply_rope(key, positions)
        # SDPA boolean masks use True=allowed, opposite to TransformerEncoderLayer.
        allowed = ~(src_mask[None, None, :, :]
                    | src_key_padding_mask[:, None, None, :])
        attended = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.dropout(self.out_projection(attended))
        feedforward = self.linear2(
            self.dropout(nn.functional.gelu(self.linear1(self.norm2(x))))
        )
        return x + self.dropout(feedforward)


class CausalTAMTransformer(nn.Module):
    name = "tam_transformer"
    seq_len_fun = staticmethod(tam_seq_len_fun)

    def __init__(
        self,
        n_classes: int,
        features: Sequence[Feats],
        trim_beginning: int,
        d_model: int = 128,
        nhead: int = 4,
        nlayer: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        max_tokens: int = 300,
        predict_ks: int = 11,
        tam_dict: dict | None = None,
        positional_encoding: str = "learned",
    ):
        super().__init__()
        allowed = {Feats.TAM_UP_COUNTS, Feats.TAM_DOWN_COUNTS,
                   Feats.TAM_TIMES, *TAM_TIME_STD_FEATURES}
        if not features or not set(features).issubset(allowed):
            raise ValueError("Transformer requires TAM count/time features")
        if min(n_classes, d_model, nhead, nlayer, dim_feedforward, max_tokens,
               predict_ks) <= 0 or d_model % nhead:
            raise ValueError(
                "Positive dimensions and d_model divisible by nhead required"
            )
        if positional_encoding not in {"learned", "rope"}:
            raise ValueError("positional_encoding must be 'learned' or 'rope'")
        if positional_encoding == "rope" and (d_model // nhead) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.features = list(features)
        self.trim_beginning = trim_beginning
        self.tam_dict = dict(tam_dict or {})
        self.predict_ks = predict_ks
        self.max_tokens = max_tokens
        self.model_config: dict[str, Any] = {
            "n_classes": n_classes, "features": self.features,
            "trim_beginning": trim_beginning, "d_model": d_model, "nhead": nhead,
            "nlayer": nlayer, "dim_feedforward": dim_feedforward,
            "dropout": dropout, "max_tokens": max_tokens,
            "predict_ks": predict_ks, "tam_dict": self.tam_dict,
            "positional_encoding": positional_encoding,
        }
        self.input_projection = nn.Linear(len(features), d_model)
        self.position = (
            nn.Embedding(max_tokens, d_model) if positional_encoding == "learned"
            else None
        )
        # Construct layers independently rather than cloning one initialization.
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            if positional_encoding == "rope" else nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(nlayer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def pack_and_forward(self, x, h, seq_lens):
        """Compatibility API: full causal sequence, no recurrent state or packing."""
        if h is not None:
            raise ValueError("Transformer has no recurrent hidden state")
        values = torch.cat(
            [cast(torch.Tensor, _feature_map(x, f)) for f in self.features], dim=-1,
        )
        if values.ndim != 3:
            raise ValueError("TAM inputs must have shape (batch, tokens)")
        bs, tokens, _ = values.shape
        if not 1 <= tokens <= self.max_tokens:
            raise ValueError(
                f"Transformer supports 1..{self.max_tokens} tokens; got {tokens}"
            )
        lengths = seq_lens.to(values.device)
        if lengths.shape != (bs,) or bool(((lengths < 1) | (lengths > tokens)).any()):
            raise ValueError("Sequence lengths must be in 1..tokens for every row")
        positions = torch.arange(tokens, device=values.device)
        padding = positions[None, :] >= lengths[:, None]
        # Remove padding before projection: padded NaNs must not enter attention.
        values = values.masked_fill(padding[..., None], 0)
        hidden = self.input_projection(values)
        if self.position is not None:
            hidden = hidden + self.position(positions)[None, :, :]
        causal_mask = torch.ones(
            tokens, tokens, device=values.device, dtype=torch.bool,
        ).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, src_key_padding_mask=padding)
        logits = self.classifier(self.norm(hidden))
        return logits.masked_fill(padding[..., None], 0), None

    def forward(self, x, h=None):
        return self.pack_and_forward(x, h, self.seq_len_fun(x))

    @torch.no_grad()
    def predict(self, x, h=None, seq_lens=None, ks=None):
        lengths = self.seq_len_fun(x) if seq_lens is None else seq_lens
        logits, _ = self.pack_and_forward(x, h, lengths)
        return select_tam_prediction(
            logits, lengths.to(logits.device), ks or self.predict_ks, mask_padding=True,
        )
