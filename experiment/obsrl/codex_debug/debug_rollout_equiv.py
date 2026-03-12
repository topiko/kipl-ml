import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from experiment.obsrl.sim import _rollout_single_pass, _rollout_streaming
from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


class StatefulPatternSelector(nn.Module):
    """Deterministic selector head for debug plots/tests.

    Keeps an internal timestep cursor so it behaves the same when called on the
    full sequence (single-pass) or step-by-step (discrete rollout).
    """

    def __init__(self, pattern: list[int], n_actions: int, high: float = 10.0):
        super().__init__()
        if n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        if len(pattern) == 0:
            raise ValueError("pattern must be non-empty")
        if any((p < 0 or p >= n_actions) for p in pattern):
            raise ValueError("pattern values must be in [0, n_actions)")
        self.pattern = pattern
        self.n_actions = int(n_actions)
        self.high = float(high)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, L, H); output must be (B, L, n_actions)
        B, L = x.shape[0], x.shape[1]
        out = torch.zeros((B, L, self.n_actions), device=x.device, dtype=x.dtype)
        idxs = torch.tensor(
            [self.pattern[(self.pos + t) % len(self.pattern)] for t in range(L)],
            device=x.device,
            dtype=torch.long,
        )
        out.scatter_(
            dim=-1,
            index=idxs.view(1, L, 1).expand(B, L, 1),
            src=torch.full((B, L, 1), self.high, device=x.device, dtype=x.dtype),
        )

        self.pos += int(L)
        return out


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
    if len(samples_s) == 0:
        return TimingStats(n=0, mean_ms=float("nan"), std_ms=float("nan"), median_ms=float("nan"), min_ms=float("nan"), max_ms=float("nan"))
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
    # For int tensors, use exact comparison
    if aa.is_floating_point():
        try:
            torch.testing.assert_close(aa, bb, atol=atol, rtol=rtol, msg=name)
        except AssertionError as e:
            diff = (aa - bb).abs()
            mx = float(diff.max().item()) if diff.numel() else 0.0
            mean = float(diff.mean().item()) if diff.numel() else 0.0
            raise AssertionError(f"{name}: max_abs_diff={mx:.6g} mean_abs_diff={mean:.6g}") from e
    else:
        if not torch.equal(aa, bb):
            diff = (aa - bb).abs()
            mx = int(diff.max().item()) if diff.numel() else 0
            raise AssertionError(f"{name}: int tensor mismatch, max_diff={mx}")


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
    parser.add_argument("--plot", action="store_true", help="Save debug plots")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures with plt.show() (requires GUI backend)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiment/obsrl/codex_debug/debug_out",
        help="Output dir for plots",
    )
    parser.add_argument(
        "--plot_idxs",
        type=str,
        default="0",
        help="Comma-separated batch indices to plot",
    )
    parser.add_argument(
        "--selector_pattern",
        type=str,
        default="do_nothing",
        choices=["do_nothing", "send_cycle", "delay_cycle", "send_and_delay_cycle"],
        help="Override action selector to force varied actions",
    )
    args = parser.parse_args()

    if args.plot or args.show:
        import matplotlib

        # Default to a headless backend unless user explicitly asks to show.
        # If there is no display, plt.show() won't work anyway.
        if not args.show and os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt  # noqa: F401

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
        prefer_wait_bias=0.0,
        enable_delay=(args.selector_pattern in {"delay_cycle", "send_and_delay_cycle"}),
        prob_eps={
            Actions.SELECTOR: 0.0,
            Actions.SEND_COUNT_UP: 0.0,
            Actions.SEND_TIME_UP: 0.0,
            Actions.SEND_COUNT_DOWN: 0.0,
            Actions.SEND_TIME_DOWN: 0.0,
        },
    ).to(device)
    obs.eval()

    if args.selector_pattern != "do_nothing":
        if args.selector_pattern == "send_cycle":
            # 0=DO_NOTHING, 1=SEND_UP, 2=SEND_DOWN, 3=SEND_BOTH
            n_actions = 4
            pattern = [0, 1, 2, 3]
        elif args.selector_pattern == "delay_cycle":
            # 0=DO_NOTHING, 4=DELAY
            n_actions = 5
            pattern = [0, 4]
        elif args.selector_pattern == "send_and_delay_cycle":
            # + 4=DELAY
            n_actions = 5
            pattern = [0, 1, 2, 3, 4]
        else:
            raise ValueError("Unknown selector_pattern")

        selector_override = StatefulPatternSelector(
            pattern=pattern, n_actions=n_actions, high=10.0
        )
        obs.actor["action_selection"] = selector_override

        # For delay-only visualizations, keep conditional heads deterministic.
        if args.selector_pattern == "delay_cycle":
            for k in ("send_count_u", "send_count_d", "send_time_u", "send_time_d"):
                lin = obs.actor[k][-1]
                if hasattr(lin, "bias"):
                    with torch.no_grad():
                        lin.bias.zero_()
                        lin.bias[0] = 10.0

    # In deterministic mode, argmax selector + argmax conditionals.
    # For do_nothing pattern, bias the selector strongly towards index 0.
    if args.selector_pattern == "do_nothing":
        obs._init_action_selection_prefer_wait(prefer_wait_bias=10.0)

    disc = DummyDisc(n_classes=5).to(device)

    reward_scales = {
        "clf_scale": 1.0,
        "d_clf_scale": 1.0,
        "padding_scale": 1.0,
        "delay_scale": 0.0,
    }
    if args.selector_pattern != "do_nothing":
        # For forced action patterns we focus on qualitative behavior/plots.
        # Reward equivalence can break due to sort tie-breaking when many packets
        # share identical timestamps.
        reward_scales = None

    if args.selector_pattern != "do_nothing":
        selector_override.reset()  # type: ignore[name-defined]

    skip_equiv = args.selector_pattern in {"delay_cycle", "send_and_delay_cycle"}

    if skip_equiv:
        # Delay is only supported for stepwise execution.
        out_a = _rollout_streaming(
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
        out_b = out_a
    else:
        out_a = _rollout_single_pass(
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

        if args.selector_pattern != "do_nothing":
            selector_override.reset()  # type: ignore[name-defined]

        out_b = _rollout_streaming(
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

    seq_lens = (fd_a[Feats.TIMES] >= 0).sum(dim=1)
    T = act_times_a.shape[1]
    mask = torch.arange(T, device=device)[None, :] < seq_lens[:, None]

    if not skip_equiv:
        _assert_close_masked(act_times_a, act_times_b, mask, "act_times", atol=1e-6)
        _assert_close_masked(log_ps_a, log_ps_b, mask, "log_ps", atol=1e-6)
        if values_a is not None or values_b is not None:
            if values_a is None or values_b is None:
                raise AssertionError("Value presence mismatch")
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

    if not skip_equiv:
        torch.testing.assert_close(
            pkt_len_a, pkt_len_b, atol=0, rtol=0, msg="X_obs.pkt_len"
        )

    def _sorted_rows(Xt: dict[Feats, torch.Tensor], i: int, n: int) -> np.ndarray:
        t = Xt[Feats.TIMES][i, :n].detach().cpu().numpy()
        d = Xt[Feats.DIRS][i, :n].detach().cpu().numpy()
        p = Xt[Feats.PADDING][i, :n].detach().cpu().numpy()
        # Round times so we can compare with deterministic ordering.
        t = np.round(t.astype(np.float64), 6)
        rows = np.stack([t, d, p], axis=1)
        # Lexicographic sort (time, dir, padding)
        order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
        return rows[order]

    if not skip_equiv:
        for i in range(Xobs_a[Feats.DIRS].shape[0]):
            n = int(pkt_len_a[i].item())
            if args.selector_pattern == "do_nothing":
                # Strict positional match for the baseline equivalence case.
                torch.testing.assert_close(
                    Xobs_a[Feats.TIMES][i, :n],
                    Xobs_b[Feats.TIMES][i, :n],
                    atol=1e-6,
                    rtol=0.0,
                    msg=f"X_obs.times[{i}]",
                )
                torch.testing.assert_close(
                    Xobs_a[Feats.DIRS][i, :n],
                    Xobs_b[Feats.DIRS][i, :n],
                    atol=0,
                    rtol=0,
                    msg=f"X_obs.dirs[{i}]",
                )
                torch.testing.assert_close(
                    Xobs_a[Feats.PADDING][i, :n],
                    Xobs_b[Feats.PADDING][i, :n],
                    atol=0,
                    rtol=0,
                    msg=f"X_obs.padding[{i}]",
                )
            else:
                # When there are many packets with identical timestamps (e.g. fixed-mode
                # sends at zero shift), the final sort tie-breaking may differ between
                # single-pass and stepwise execution. Compare as a multiset instead.
                ra = _sorted_rows(Xobs_a, i, n)
                rb = _sorted_rows(Xobs_b, i, n)
                if ra.shape != rb.shape or not np.array_equal(ra, rb):
                    raise AssertionError(f"X_obs multiset mismatch for batch {i}")

    if not skip_equiv and (rewards_a is not None or rewards_b is not None):
        if rewards_a is None or rewards_b is None:
            raise AssertionError("Reward presence mismatch")

        # Rewards are (nleague, B, T). Compare league 0 for active windows.
        rmask = mask.unsqueeze(0).expand_as(next(iter(rewards_a.values())))
        for k in rewards_a.keys():
            _assert_close_masked(rewards_a[k], rewards_b[k], rmask, f"rewards.{k}", atol=1e-6)

    if skip_equiv:
        print("OK: produced discrete(streaming) rollout with delay")
    else:
        print("OK: discrete(streaming) matches single-pass (deterministic act)")

    if args.plot or args.show:
        import matplotlib.pyplot as plt
        from kipl_ml.tools.plottr import plot_actions, plot_obs_features, plot_trace

        os.makedirs(args.outdir, exist_ok=True)
        idxs = [int(s) for s in args.plot_idxs.split(",") if s.strip()]
        for i in idxs:
            fig, axes = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
            fig.suptitle(
                f"rollout equiv debug (batch={i}) pattern={args.selector_pattern}"
            )

            plot_trace({k: v.clone() for k, v in X.items()}, idx=i, ax=axes[0])
            axes[0].set_title("Base trace")

            plot_trace(Xobs_a, idx=i, ax=axes[1])
            axes[1].set_title(
                "X_obs (single-pass)" if not skip_equiv else "X_obs (discrete+streaming)"
            )

            plot_trace(Xobs_b, idx=i, ax=axes[2])
            axes[2].set_title("X_obs (discrete+streaming)")

            plot_obs_features(fd_a, idx=i, ax=axes[3])
            axes[3].set_title("Obs window features")

            plot_actions(act_times_a, actions_a, idx=i, ax=axes[4])
            axes[4].set_title("Actions (includes DELAY spans if present)")

            out_path = os.path.join(
                args.outdir,
                f"rollout_equiv_{args.selector_pattern}_batch{i:03d}.png",
            )
            fig.tight_layout()
            if args.plot:
                fig.savefig(out_path, dpi=160)
                print("wrote", out_path)
            if args.show:
                plt.show()
            plt.close(fig)

    # Timing
    # Note: _rollout_single_pass currently mutates X in get_window_feature_dict(extend_end_s=2),
    # so we must clone inputs for each timed call.
    def _call_single_pass(rewards: bool) -> None:
        if args.selector_pattern != "do_nothing":
            selector_override.reset()  # type: ignore[name-defined]
        _rollout_single_pass(
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
        if args.selector_pattern != "do_nothing":
            selector_override.reset()  # type: ignore[name-defined]
        _rollout_streaming(
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

    if args.runs > 0 or args.runs_rewards > 0:
        with torch.no_grad():
            if args.runs > 0:
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
            else:
                t_single = _timing_stats([])
                t_disc = _timing_stats([])

            if args.runs_rewards > 0:
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
            else:
                t_single_r = _timing_stats([])
                t_disc_r = _timing_stats([])

        if t_single.n > 0 and t_disc.n > 0:
            ratio = t_disc.mean_ms / max(t_single.mean_ms, 1e-12)
        else:
            ratio = float("nan")
        if t_single_r.n > 0 and t_disc_r.n > 0:
            ratio_r = t_disc_r.mean_ms / max(t_single_r.mean_ms, 1e-12)
        else:
            ratio_r = float("nan")

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
