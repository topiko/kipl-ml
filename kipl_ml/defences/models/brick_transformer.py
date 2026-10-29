"""Bounded causal history encoder for brick selection."""

import torch
from torch import nn

from kipl_ml.defences.models.tam_transformer import RoPEEncoderLayer


class BrickTransformer(nn.Module):
    """Shared attention weights, separate side inputs; no cross-side attention.

    State contains scaled observation/brick tokens, not projected features or KV
    caches. Recomputing the small rolling window keeps training and replay equal
    and gives gradients to projections of every retained token.
    """

    def __init__(
        self, n_features: int, n_client: int, n_server: int, width: int,
        nhead: int = 2, nlayer: int = 2, max_tokens: int = 32,
    ):
        super().__init__()
        if min(width, nhead, nlayer, max_tokens) <= 0 or width % nhead:
            raise ValueError(
                "positive dimensions and width divisible by nhead required"
            )
        if (width // nhead) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.n_features = n_features
        self.n_client = n_client
        self.max_tokens = max_tokens
        self.client_projection = nn.Linear(n_features + n_client, width)
        self.server_projection = nn.Linear(n_features + n_server, width)
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(width, nhead, 2 * width, dropout=0.)
            for _ in range(nlayer)
        ])
        self.norm = nn.LayerNorm(width)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Encode (B,T,features+client+server) as (2B,T,H), clients then servers."""
        boundary = self.n_features + self.n_client
        client = self.client_projection(history[..., :boundary])
        server = self.server_projection(torch.cat((
            history[..., :self.n_features], history[..., boundary:],
        ), dim=-1))
        hidden = torch.cat((client, server), dim=0)
        batch, tokens, _ = hidden.shape
        causal = torch.ones(
            tokens, tokens, device=hidden.device, dtype=torch.bool,
        ).triu(1)
        padding = torch.zeros(batch, tokens, device=hidden.device, dtype=torch.bool)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal, src_key_padding_mask=padding)
        return self.norm(hidden)

    def forward(
        self, features: torch.Tensor, client_brick: torch.Tensor,
        server_brick: torch.Tensor, state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = torch.cat((features, client_brick, server_brick), dim=-1).unsqueeze(1)
        history = token if state is None else torch.cat((state, token), dim=1)
        history = history[:, -self.max_tokens:]
        client_h, server_h = self.encode(history)[:, -1].chunk(2, dim=0)
        return client_h, server_h, history
