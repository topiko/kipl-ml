import torch


def get_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    _, T = rewards.shape
    G = torch.zeros_like(rewards)
    R = 0

    # for r in rewards.flip(dims=(1,)).T:
    for t in reversed(range(T)):
        R = rewards[:, t] + gamma * R
        G[:, t] = R
    return G


def get_gae(
    rewards: torch.Tensor, values: torch.Tensor, lambda_: float, gamma: float
) -> torch.Tensor:
    """
    rewards, values: [B, T]
    returns: GAE advantages of shape [B, T]
    """

    B, T = rewards.shape
    # delta = r_t + gamma * V_t+1 - V_t
    deltas = torch.zeros_like(rewards)
    deltas[:, :-1] = rewards[:, :-1] + gamma * values[:, 1:] - values[:, :-1]
    # V_t+1 for last times step = 0
    deltas[:, -1] = rewards[:, -1] - values[:, -1]

    gae = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device)

    y = lambda_ * gamma
    for t in reversed(range(T)):
        running = deltas[:, t] + running * y
        gae[:, t] = running

    return gae
