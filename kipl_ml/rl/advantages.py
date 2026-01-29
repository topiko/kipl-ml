import torch


def get_returns(
    rewards: torch.Tensor,
    seq_lens: torch.Tensor,
    gamma: float,
    bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T = rewards.shape

    # (B, T)
    mask = torch.arange(T, device=rewards.device)[None, :] < seq_lens[:, None]
    mask_f = mask.to(rewards.dtype)

    if bootstrap is None:
        bootstrap = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    else:
        bootstrap = bootstrap.to(device=rewards.device, dtype=rewards.dtype)

    G = torch.zeros_like(rewards)
    R = bootstrap.clone()
    for t in reversed(range(T)):
        R = torch.where(mask[:, t], rewards[:, t] + gamma * R, bootstrap)
        G[:, t] = R * mask_f[:, t]  # cut off at end

    return G


def get_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    seq_lens: torch.Tensor,
    lambda_: float,
    gamma: float,
    bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    if bootstrap is not None:
        raise NotImplementedError()

    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    # (B, T) mask where valid steps are 1
    mask = (torch.arange(T, device=device)[None, :] < seq_lens[:, None]).to(dtype)

    # next_mask[t] is 1 iff t+1 is valid (so we can bootstrap/propagate)
    next_mask = torch.zeros_like(mask)
    next_mask[:, :-1] = mask[:, 1:]

    # delta_t = r_t + gamma * V_{t+1} * next_mask_t - V_t
    # last col will be ignored due to next_mask=0
    values_next = values.roll(shifts=-1, dims=1)
    deltas = rewards + gamma * values_next * next_mask - values
    deltas = deltas * mask  # keep padding clean

    gae = torch.zeros_like(rewards)
    running = torch.zeros(B, device=device, dtype=dtype)
    y = lambda_ * gamma

    for t in reversed(range(T)):
        # stop propagation past end
        running = deltas[:, t] + running * y * next_mask[:, t]
        gae[:, t] = running * mask[:, t]

    return gae
