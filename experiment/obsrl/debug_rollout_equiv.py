import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from experiment.obsrl.sim import rollout, rollout_discrete_streaming
from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


class DummyDisc(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.n_classes = n_classes
        self.feat_mode = "dir"

    def seq_len_fun(self, x: dict[Feats, torch.Tensor]) -> torch.Tensor:
        return (x[Feats.DIRS] != 0).sum(dim=1)

    def pack_and_forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: None,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # (B, N, C)
        B, N = x[Feats.DIRS].shape
        logits = torch.zeros((B, N, self.n_classes), device=x[Feats.DIRS].device)
        return logits, None


def _make_synth_batch(
    bs: int,
    n_packets: int,
    dt: float,
    device: torch.device,
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
    rng = np.random.default_rng(0)
    seq_lens = rng.integers(low=int(0.6 * n_packets), high=n_packets, size=bs)

    times = torch.zeros((bs, n_packets), device=device)
    dirs = torch.zeros((bs, n_packets), device=device)
    padding = torch.zeros((bs, n_packets), device=device)

    for i in range(bs):
        L = int(seq_lens[i])
        iats = rng.uniform(low=0.2 * dt, high=1.5 * dt, size=L)
        t = np.cumsum(iats)
        times[i, :L] = torch.tensor(t, device=device, dtype=torch.float32)
        times[i, L:] = times[i, L - 1]

        d = rng.choice([UPLOAD, DOWNLOAD], size=L)
        dirs[i, :L] = torch.tensor(d, device=device, dtype=torch.float32)
        dirs[i, L:] = 0
        padding[i, :] = 0

    X = {
        Feats.TIMES: times,
        Feats.DIRS: dirs,
        Feats.PADDING: padding,
    }
    # labels in [0..4]
    y = torch.arange(bs, device=device) % 5
    return X, y


@dataclass
class TimingStats:
    n: int
    mean_ms: float
    std_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


def _timing_stats(samples_s: list[float]) -> TimingStats:
    xs = np.asarray(samples_s, dtype=np.float64)
    return TimingStats(
        n=int(xs.size),
        mean_ms=float(xs.mean() * 1e3),
        std_ms=float(xs.std(ddof=0) * 1e3),
        median_ms=float(np.median(xs) * 1e3),
        min_ms=float(xs.min() * 1e3),
        max_ms=float(xs.max() * 1e3),
    )


def _timeit(
    fn,
    *,
    warmup: int,
    runs: int,
    device: torch.device,
) -> TimingStats:
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        fn()
    _sync()

    samples: list[float] = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        samples.append(t1 - t0)

    return _timing_stats(samples)


def _assert_close_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    name: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    aa = a[mask]
    bb = b[mask]
    try:
        torch.testing.assert_close(aa, bb, atol=atol, rtol=rtol, msg=name)
    except AssertionError as e:
        diff = (aa - bb).abs()
        mx = float(diff.max().item()) if diff.numel() else 0.0
        mean = float(diff.mean().item()) if diff.numel() else 0.0
        raise AssertionError(f"{name}: max_abs_diff={mx:.6g} mean_abs_diff={mean:.6g}") from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--runs_rewards",
        type=int,
        default=5,
        help="Timing runs when reward computation enabled",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(int(args.threads))

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(args.device)
    dt = 0.02

    X, y = _make_synth_batch(bs=6, n_packets=200, dt=dt, device=device)

    obs = AGENT1(
        time_step=dt,
        max_silence_s=0.2,
        hsize=64,
        nlayers=2,
        send_mode="fixed",
        prefer_wait_bias=10.0,
        prob_eps={
            Actions.SELECTOR: 0.0,
            Actions.SEND_COUNT_UP: 0.0,
            Actions.SEND_TIME_UP: 0.0,
            Actions.SEND_COUNT_DOWN: 0.0,
            Actions.SEND_TIME_DOWN: 0.0,
        },
    ).to(device)
    obs.eval()

    disc = DummyDisc(n_classes=5).to(device)

    reward_scales = {
        "clf_scale": 1.0,
        "d_clf_scale": 1.0,
        "padding_scale": 1.0,
    }

    out_a = rollout(
        obs=obs,
        critic=None,
        disc=disc,
        X={k: v.clone() for k, v in X.items()},
        y=y.clone(),
        disc_league=[(0, None)],
        disc_features=None,
        detach_period=50,
        critic_detach_period=None,
        reward_scales=reward_scales,
        sample=False,
    )

    out_b = rollout_discrete_streaming(
        obs=obs,
        critic=None,
        disc=disc,
        X={k: v.clone() for k, v in X.items()},
        y=y.clone(),
        disc_league=[(0, None)],
        disc_features=None,
        detach_period=50,
        critic_detach_period=None,
        reward_scales=reward_scales,
        sample=False,
    )

    (
        log_ps_a,
        sel_probs_a,
        values_a,
        rewards_a,
        ent_a,
        act_times_a,
        actions_a,
        Xobs_a,
        fd_a,
    ) = out_a
    (
        log_ps_b,
        sel_probs_b,
        values_b,
        rewards_b,
        ent_b,
        act_times_b,
        actions_b,
        Xobs_b,
        fd_b,
    ) = out_b

    seq_lens = fd_a[Feats.TIMES].isnan().logical_not().sum(dim=1)
    T = act_times_a.shape[1]
    mask = torch.arange(T, device=device)[None, :] < seq_lens[:, None]

    _assert_close_masked(act_times_a, act_times_b, mask, "act_times", atol=1e-6)
    _assert_close_masked(log_ps_a, log_ps_b, mask, "log_ps", atol=1e-6)
    _assert_close_masked(values_a, values_b, mask, "values", atol=1e-6)
    _assert_close_masked(
        ent_a["selection_entropy"],
        ent_b["selection_entropy"],
        mask,
        "entropy.selection",
        atol=1e-6,
    )
    _assert_close_masked(
        ent_a["conditional_entropy"],
        ent_b["conditional_entropy"],
        mask,
        "entropy.conditional",
        atol=1e-6,
    )

    for k in actions_a.keys():
        _assert_close_masked(actions_a[k], actions_b[k], mask, f"actions.{k}")

    # sel_probs has an extra action dimension.
    probs_mask = mask.unsqueeze(-1).expand_as(sel_probs_a)
    _assert_close_masked(sel_probs_a, sel_probs_b, probs_mask, "sel_probs", atol=1e-6)

    # Compare realized packets only (DIRS!=0). Tail padding TIMES can differ
    # between single-pass vs stepwise send_exec due to how send_exec fills
    # post-seq-end times.
    pkt_len_a = (Xobs_a[Feats.DIRS] != 0).sum(dim=1)
    pkt_len_b = (Xobs_b[Feats.DIRS] != 0).sum(dim=1)
    torch.testing.assert_close(pkt_len_a, pkt_len_b, atol=0, rtol=0, msg="Xobs.pkt_len")

    for i in range(Xobs_a[Feats.DIRS].shape[0]):
        n = int(pkt_len_a[i].item())
        torch.testing.assert_close(
            Xobs_a[Feats.TIMES][i, :n],
            Xobs_b[Feats.TIMES][i, :n],
            atol=1e-12,
            rtol=0.0,
            msg=f"Xobs.times[{i}]",
        )
        torch.testing.assert_close(
            Xobs_a[Feats.DIRS][i, :n],
            Xobs_b[Feats.DIRS][i, :n],
            atol=0,
            rtol=0,
            msg=f"Xobs.dirs[{i}]",
        )
        torch.testing.assert_close(
            Xobs_a[Feats.PADDING][i, :n],
            Xobs_b[Feats.PADDING][i, :n],
            atol=0,
            rtol=0,
            msg=f"Xobs.padding[{i}]",
        )

    if rewards_a is None or rewards_b is None:
        raise AssertionError("Expected rewards to be present")

    # Rewards are (nleague, B, T). Compare league 0 for active windows.
    rmask = mask.unsqueeze(0).expand_as(next(iter(rewards_a.values())))
    for k in rewards_a.keys():
        _assert_close_masked(rewards_a[k], rewards_b[k], rmask, f"rewards.{k}", atol=1e-6)

    print("OK: discrete(streaming) matches single-pass (deterministic act)")

    # Timing
    # Note: rollout() currently mutates X in get_window_feature_dict(extend_end_s=2),
    # so we must clone inputs for each timed call.
    def _call_single_pass(rewards: bool) -> None:
        rollout(
            obs=obs,
            critic=None,
            disc=disc,
            X={k: v.clone() for k, v in X.items()},
            y=y.clone(),
            disc_league=[(0, None)],
            disc_features=None,
            detach_period=50,
            critic_detach_period=None,
            reward_scales=reward_scales if rewards else None,
            sample=False,
        )

    def _call_discrete(rewards: bool) -> None:
        rollout_discrete_streaming(
            obs=obs,
            critic=None,
            disc=disc,
            X={k: v.clone() for k, v in X.items()},
            y=y.clone(),
            disc_league=[(0, None)],
            disc_features=None,
            detach_period=50,
            critic_detach_period=None,
            reward_scales=reward_scales if rewards else None,
            sample=False,
        )

    with torch.no_grad():
        t_single = _timeit(
            lambda: _call_single_pass(False),
            warmup=args.warmup,
            runs=args.runs,
            device=device,
        )
        t_disc = _timeit(
            lambda: _call_discrete(False),
            warmup=args.warmup,
            runs=args.runs,
            device=device,
        )

        t_single_r = _timeit(
            lambda: _call_single_pass(True),
            warmup=args.warmup,
            runs=args.runs_rewards,
            device=device,
        )
        t_disc_r = _timeit(
            lambda: _call_discrete(True),
            warmup=args.warmup,
            runs=args.runs_rewards,
            device=device,
        )

    ratio = t_disc.mean_ms / max(t_single.mean_ms, 1e-12)
    ratio_r = t_disc_r.mean_ms / max(t_single_r.mean_ms, 1e-12)
    print(
        "Timing (includes X/y clone cost per call):\n"
        + f"- single-pass (policy): mean={t_single.mean_ms:.3f}ms std={t_single.std_ms:.3f}ms "
        + f"median={t_single.median_ms:.3f}ms min={t_single.min_ms:.3f}ms max={t_single.max_ms:.3f}ms (n={t_single.n})\n"
        + f"- discrete    (policy): mean={t_disc.mean_ms:.3f}ms std={t_disc.std_ms:.3f}ms "
        + f"median={t_disc.median_ms:.3f}ms min={t_disc.min_ms:.3f}ms max={t_disc.max_ms:.3f}ms (n={t_disc.n})\n"
        + f"- ratio(policy)={ratio:.3f}x\n"
        + f"- single-pass (rewards): mean={t_single_r.mean_ms:.3f}ms std={t_single_r.std_ms:.3f}ms "
        + f"median={t_single_r.median_ms:.3f}ms min={t_single_r.min_ms:.3f}ms max={t_single_r.max_ms:.3f}ms (n={t_single_r.n})\n"
        + f"- discrete    (rewards): mean={t_disc_r.mean_ms:.3f}ms std={t_disc_r.std_ms:.3f}ms "
        + f"median={t_disc_r.median_ms:.3f}ms min={t_disc_r.min_ms:.3f}ms max={t_disc_r.max_ms:.3f}ms (n={t_disc_r.n})\n"
        + f"- ratio(rewards)={ratio_r:.3f}x"
    )


if __name__ == "__main__":
    main()
