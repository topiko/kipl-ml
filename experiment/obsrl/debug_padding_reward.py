"""Debug padding reward computation."""

import torch
import numpy as np

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.simulate import policy_rollout_streaming
from kipl_ml.trace.enums import Feats
from kipl_ml.rl.enums import Actions


def make_synth_trace(
    n_packets: int = 5000,
    dt: float = 0.02,
    padding_frac: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> dict[Feats, torch.Tensor]:
    """Create a synthetic trace with padding packets distributed throughout."""
    n_real = int(n_packets * (1 - padding_frac))
    n_pad = n_packets - n_real

    times = torch.zeros((1, n_packets), device=device)
    dirs = torch.zeros((1, n_packets), device=device)
    padding = torch.zeros((1, n_packets), dtype=torch.bool, device=device)

    # Real packets: random inter-arrival times
    rng = np.random.default_rng(42)
    iats = rng.uniform(low=0.5 * dt, high=2.0 * dt, size=n_real)
    t = np.cumsum(iats)
    times[0, :n_real] = torch.tensor(t, device=device, dtype=torch.float32)

    # Random directions
    d = rng.choice([UPLOAD, DOWNLOAD], size=n_real)
    dirs[0, :n_real] = torch.tensor(d, device=device, dtype=torch.float32)

    # Padding packets: interspersed throughout the trace
    # Place padding packets at regular intervals
    pad_interval = n_real // n_pad if n_pad > 0 else n_real
    pad_idx = 0
    for i in range(n_real):
        if pad_idx < n_pad and i > 0 and i % pad_interval == 0:
            # Insert padding packet at this position
            times[0, n_real + pad_idx] = times[0, i]  # Same time as real packet
            dirs[0, n_real + pad_idx] = UPLOAD
            padding[0, n_real + pad_idx] = True
            pad_idx += 1

    # Sort by time
    sort_idx = torch.argsort(times[0])
    times = times[:, sort_idx]
    dirs = dirs[:, sort_idx]
    padding = padding[:, sort_idx]

    return {
        Feats.TIMES: times,
        Feats.DIRS: dirs,
        Feats.DECOY: padding,
    }


def main():
    device = torch.device("cpu")
    dt = 0.02  # 20ms time step
    n_packets = 5000
    padding_frac = 0.3

    print(f"=== Padding Reward Debug ===")
    print(f"dt={dt}s, n_packets={n_packets}, padding_frac={padding_frac}")

    # Create trace
    X = make_synth_trace(n_packets, dt, padding_frac, device)

    n_pad = X[Feats.DECOY].sum().item()
    print(f"Padding packets: {n_pad}")
    print(
        f"Trace duration: {X[Feats.TIMES][0, X[Feats.DIRS][0] != 0].max().item():.2f}s"
    )

    # Create agent
    agent = AGENT1(
        time_step=dt,
        max_silence_s=0.1,
        hsize=32,
        nlayers=1,
        prob_eps=0.0,
        enable_delay=False,
    )
    agent.eval()

    # Run streaming rollout
    print("\nRunning streaming rollout...")
    fd, act_time_bins, actions, log_ps, sel_probs, values, entropies, X_obs = (
        policy_rollout_streaming(agent, X, sample=False)
    )

    # Check outputs
    T = act_time_bins.shape[1]
    print(f"\nNumber of action steps: {T}")
    print(f"act_time_bins shape: {act_time_bins.shape}")

    # Check valid action times
    valid_mask = act_time_bins[0] >= 0
    valid_bins = act_time_bins[0, valid_mask]
    print(f"Valid action bins: {valid_bins.numel()}")
    print(f"First 5 valid bins: {valid_bins[:5].tolist()}")
    print(f"Last 5 valid bins: {valid_bins[-5:].tolist()}")

    # Check the observation windows
    print(f"\n=== Checking observation windows ===")
    time_bins_fd = fd[Feats.TIME_BINS][0]
    dt_bins_fd = fd[Feats.Dt_BINS][0]

    valid_fd_mask = time_bins_fd >= 0
    valid_time_bins = time_bins_fd[valid_fd_mask]
    valid_dt_bins = dt_bins_fd[valid_fd_mask]

    print(f"First 5 TIME_BINS: {valid_time_bins[:5].tolist()}")
    print(f"First 5 Dt_BINS: {valid_dt_bins[:5].tolist()}")
    print(f"Last 5 TIME_BINS: {valid_time_bins[-5:].tolist()}")
    print(f"Last 5 Dt_BINS: {valid_dt_bins[-5:].tolist()}")

    # Check action times = TIME_BINS + Dt_BINS
    action_bins_computed = valid_time_bins + valid_dt_bins
    print(
        f"\nFirst 5 action bins (TIME_BINS + Dt_BINS): {action_bins_computed[:5].tolist()}"
    )
    print(
        f"Last 5 action bins (TIME_BINS + Dt_BINS): {action_bins_computed[-5:].tolist()}"
    )

    # Check if there's an off-by-one error
    print(f"\n=== Checking for off-by-one ===")
    print(f"Expected: action_bins[i] = TIME_BINS[i+1] (right edge of window i)")
    print(
        f"Actual: action_bins[0] = {action_bins_computed[0].item()}, TIME_BINS[1] = {valid_time_bins[1].item() if len(valid_time_bins) > 1 else 'N/A'}"
    )

    # Check the last window
    print(
        f"\nLast window: TIME_BINS={valid_time_bins[-1].item()}, Dt_BINS={valid_dt_bins[-1].item()}"
    )
    print(f"Last action bin: {action_bins_computed[-1].item()}")
    print(
        f"Expected last action bin: {valid_time_bins[-1].item() + 1} (TIME_BINS[last] + 1)"
    )

    # Check X_obs
    print(f"\nX_obs TIMES shape: {X_obs[Feats.TIMES].shape}")
    n_pad_obs = X_obs[Feats.DECOY].sum().item()
    print(f"Padding packets in X_obs: {n_pad_obs}")

    # === KEY ISSUE: N from disc_logits vs packet count ===
    print(f"\n=== KEY ISSUE: N from disc_logits vs packet count ===")

    # Simulate what N would be for different discriminator configs
    # N = disc_logits.shape[1] is the number of TAM bins, not packets!

    # If discriminator has 100 TAM bins (2 seconds with dt=0.02s)
    N_tam_bins = 100
    print(f"\nIf N (disc_logits.shape[1]) = {N_tam_bins} TAM bins:")
    print(f"  - This covers {N_tam_bins * dt:.2f} seconds")
    print(
        f"  - But we slice X_obs[:, :{N_tam_bins}] which is only {N_tam_bins} packets!"
    )
    print(
        f"  - First {N_tam_bins} packets cover ~{X_obs[Feats.TIMES][0, :N_tam_bins].max().item():.3f} seconds"
    )

    # If discriminator has 500 TAM bins (10 seconds)
    N_tam_bins = 500
    print(f"\nIf N (disc_logits.shape[1]) = {N_tam_bins} TAM bins:")
    print(f"  - This covers {N_tam_bins * dt:.2f} seconds")
    print(
        f"  - But we slice X_obs[:, :{N_tam_bins}] which is only {N_tam_bins} packets!"
    )
    print(
        f"  - First {N_tam_bins} packets cover ~{X_obs[Feats.TIMES][0, :N_tam_bins].max().item():.3f} seconds"
    )

    # The fix: use packet_seq_lens instead of N for padding computation
    print(f"\n=== THE BUG ===")
    print(f"Padding penalty uses N = disc_logits.shape[1] = number of TAM bins")
    print(f"But it slices X_obs[:, :N] which treats N as number of packets!")
    print(
        f"This causes us to only look at the first N packets, not all packets in the first N TAM bins."
    )

    # Show the correct approach
    print(f"\n=== CORRECT APPROACH ===")
    print(f"Should use packet_seq_lens to determine how many packets to consider.")
    print(f"Or compute which packets fall within the first N TAM bins.")


if __name__ == "__main__":
    main()
