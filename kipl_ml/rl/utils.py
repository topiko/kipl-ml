from __future__ import annotations

import torch

from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def _fill_after_seq_end(
    values: torch.Tensor, pad_val: float, fill_val: str = "nan"
) -> torch.Tensor:
    mask = (values != pad_val).float()

    rows, cols = torch.where(mask.diff(dim=1) < 0)

    # If padding values appear in multiple segments within a row, fall back to
    # using the first detected seq-end boundary.
    if rows.numel() > 0:
        # Compute the first boundary column per row.
        first_col = torch.full(
            (values.shape[0],),
            values.shape[1],
            device=values.device,
            dtype=cols.dtype,
        )
        first_col.scatter_reduce_(0, rows, cols, reduce="amin", include_self=True)

        rows_u = rows.unique()
        rows = rows_u
        cols = first_col[rows_u]

    for idx_r, idx_c in zip(rows, cols):
        if fill_val == "nan":
            values[idx_r, (idx_c + 1) :] = torch.nan
        elif fill_val == "last":
            values[idx_r, (idx_c + 1) :] = values[idx_r, idx_c]
        elif fill_val == "max":
            values[idx_r, (idx_c + 1) :] = values[idx_r, :].max()
        else:
            raise ValueError(f"Unknown fill_val option: {fill_val}")

    return values


def fill_after_seq_end(
    values: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    fill_val: str = "nan",
) -> torch.Tensor:
    """Fill values after seq end using an explicit mask.

    This avoids relying on a sentinel pad value (e.g. 0), which is ambiguous for
    legitimate features like TIMES where 0 can be a real value.
    """

    if values.shape != keep_mask.shape:
        raise ValueError(
            f"values and keep_mask must match, got {values.shape} and {keep_mask.shape}"
        )

    out = values.clone()
    B, L = out.shape
    seq_lens = keep_mask.sum(dim=1).long()

    if fill_val == "nan":
        fill = torch.full((B,), torch.nan, device=out.device, dtype=out.dtype)
    elif fill_val == "last":
        idx = (seq_lens - 1).clamp(min=0)
        fill = out.gather(1, idx.view(B, 1)).squeeze(1)
    elif fill_val == "max":
        masked = torch.where(
            keep_mask,
            out,
            torch.full((), -torch.inf, device=out.device, dtype=out.dtype),
        )
        fill = masked.max(dim=1).values
        fill = torch.where(seq_lens > 0, fill, torch.zeros_like(fill))
    else:
        raise ValueError(f"Unknown fill_val option: {fill_val}")

    col = torch.arange(L, device=out.device).unsqueeze(0).expand(B, -1)
    tail = col >= seq_lens.unsqueeze(1)
    out[tail] = fill.unsqueeze(1).expand_as(out)[tail]
    return out


def _flush_left(
    values: torch.Tensor, keep_mask: torch.Tensor, pad_val: float = 0
) -> torch.Tensor:
    if values.shape != keep_mask.shape:
        raise ValueError(
            f"Size of values and mask must match got: {values.shape} and {keep_mask.shape}."
        )

    B = values.shape[0]

    # (B, )
    values_pushed = torch.ones_like(values) * pad_val

    # (B, M) row indices, M = values.shape[1]
    row_idxs = (
        torch.arange(B, device=values.device).unsqueeze(1).expand(-1, values.shape[1])
    )

    col_idxs = keep_mask.cumsum(dim=1) - 1

    # (N, )
    row_valid = row_idxs[keep_mask]
    col_valid = col_idxs[keep_mask]

    values_pushed[row_valid, col_valid] = values[keep_mask]

    return values_pushed


def _append_values(
    base: torch.Tensor, values: torch.Tensor, on_short_base: str = "raise"
) -> torch.Tensor:
    if base.shape[0] != values.shape[0]:
        raise ValueError("Batch size of buffer and values must match")

    B = base.shape[0]

    # (B, )
    start_idxs = (base != 0).sum(dim=1)

    if ((start_idxs + values.shape[1]) > base.shape[1]).any():
        if on_short_base == "raise":
            raise ValueError("Base buffer too short to append values")
        elif on_short_base == "cat":
            # Allow for base expansion
            base = torch.cat(
                (base, torch.zeros((B, values.shape[1]), device=base.device)), dim=1
            )
        else:
            raise ValueError(f"Unknown on_short_base option: {on_short_base}")

    M = values.shape[1]
    # (B, M) column indices
    col_idxs = torch.arange(M, device=base.device).unsqueeze(0).expand(
        B, -1
    ) + start_idxs.unsqueeze(1)

    # (B, M) row indices
    row_idxs = torch.arange(B, device=base.device).unsqueeze(1).expand(-1, M)

    base[row_idxs, col_idxs] = values

    return base
