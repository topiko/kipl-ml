import numpy as np
import torch

import argparse
import os

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.observation import WindowFeatureStreamer, get_window_feature_dict
from kipl_ml.trace.enums import Feats


def _make_synth_batch(
    bs: int,
    n_packets: int,
    dt: float,
    device: torch.device,
) -> dict[Feats, torch.Tensor]:
    rng = np.random.default_rng(0)
    seq_lens = rng.integers(low=max(1, int(0.6 * n_packets)), high=n_packets, size=bs)

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

    return {
        Feats.TIMES: times,
        Feats.DIRS: dirs,
        Feats.PADDING: padding,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
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
    parser.add_argument("--idx", type=int, default=0, help="Batch index to plot")
    args = parser.parse_args()

    if args.plot or args.show:
        import matplotlib

        if not args.show and os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt  # noqa: F401

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    dt = 0.02
    max_silence_s = 0.2
    extend_end_s = 2.0

    X = _make_synth_batch(bs=8, n_packets=600, dt=dt, device=device)
    features = [Feats.UP_COUNT, Feats.DOWN_COUNT, Feats.Dt_BINS, Feats.TIME_BINS, Feats.SILENCE_FLAG]

    fd_full = get_window_feature_dict(
        {k: v.clone() for k, v in X.items()},
        dt,
        max_silence_s,
        features=features,
        extend_end_s=extend_end_s,
    )

    streamer = WindowFeatureStreamer(
        X,
        dt=dt,
        max_silence_s=max_silence_s,
        features=features,
        extend_end_s=extend_end_s,
    )

    steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in features}

    # Safety bound: with silence insertion, we can have more windows than bins.
    # Worst case: one window per bin + silence windows. Use 3x the bin count.
    max_bins = int((X[Feats.TIMES].max().item() + extend_end_s) / dt) + 10
    max_steps = max_bins * 3

    for _ in range(max_steps):
        fd_t = streamer.step()
        for f in features:
            steps[f].append(fd_t[f])
        if streamer.done.all():
            break
    else:
        raise RuntimeError("Streamer did not finish within max_steps")

    fd_stream = {f: torch.cat(steps[f], dim=1) for f in features}

    # Compare lengths.
    seq_lens_full = fd_full[Feats.SEQ_LENS]
    # TIMES are int bins with -1 sentinel; count valid entries.
    seq_lens_stream = (fd_stream[Feats.TIME_BINS] >= 0).sum(dim=1)
    torch.testing.assert_close(seq_lens_stream, seq_lens_full, atol=0, rtol=0)

    # Ensure same max length.
    L_full = fd_full[Feats.TIME_BINS].shape[1]
    L_stream = fd_stream[Feats.TIME_BINS].shape[1]
    if L_stream != L_full:
        raise AssertionError(f"Length mismatch: stream={L_stream} full={L_full}")

    # Exact compare on valid positions (TIMES >= 0 for int bins).
    mask = fd_full[Feats.TIME_BINS] >= 0
    for f in features:
        a = fd_full[f]
        b = fd_stream[f]
        atol = 0.0
        if f in {Feats.Dt_BINS, Feats.TIME_BINS}:
            atol = 0  # int bins, exact match
        try:
            torch.testing.assert_close(
                a[mask], b[mask], atol=atol, rtol=0.0, msg=str(f)
            )
        except AssertionError as e:
            diff = (a - b).abs()
            diff = torch.where(mask, diff, torch.zeros_like(diff))
            mx = diff.max().item() if diff.numel() else 0.0
            idx = torch.nonzero(diff == diff.max(), as_tuple=False)
            if idx.numel() >= 2:
                i0, t0 = int(idx[0, 0].item()), int(idx[0, 1].item())
                print(f"Mismatch in {f}: max_abs_diff={mx} at row={i0}, col={t0}")
                print("full.times", fd_full[Feats.TIME_BINS][i0, :10])
                print("stream.times", fd_stream[Feats.TIME_BINS][i0, :10])
                print("full.dt", fd_full[Feats.Dt_BINS][i0, :10])
                print("stream.dt", fd_stream[Feats.Dt_BINS][i0, :10])
            raise e

        # -1 sentinel in the tail should align too (for int-bin features).
        if f in (Feats.TIME_BINS, Feats.Dt_BINS):
            if not torch.equal(a < 0, b < 0):
                raise AssertionError(f"Sentinel mask mismatch for {f}")

    print("OK: WindowFeatureStreamer matches get_window_feature_dict")

    if args.plot or args.show:
        import matplotlib.pyplot as plt

        os.makedirs(args.outdir, exist_ok=True)
        i = int(args.idx)
        mask_i = fd_full[Feats.TIME_BINS][i] >= 0

        fig, axes = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
        fig.suptitle(f"WindowFeatureStreamer equiv (batch={i})")

        t = fd_full[Feats.TIME_BINS][i][mask_i].cpu().numpy()
        up_full = fd_full[Feats.UP_COUNT][i][mask_i].cpu().numpy()
        up_stream = fd_stream[Feats.UP_COUNT][i][mask_i].cpu().numpy()
        down_full = fd_full[Feats.DOWN_COUNT][i][mask_i].cpu().numpy()
        down_stream = fd_stream[Feats.DOWN_COUNT][i][mask_i].cpu().numpy()
        dt_full = fd_full[Feats.Dt_BINS][i][mask_i].cpu().numpy()
        dt_stream = fd_stream[Feats.Dt_BINS][i][mask_i].cpu().numpy()
        sil_full = fd_full[Feats.SILENCE_FLAG][i][mask_i].cpu().numpy()
        sil_stream = fd_stream[Feats.SILENCE_FLAG][i][mask_i].cpu().numpy()

        axes[0].plot(t, up_full, label="up_full", lw=1)
        axes[0].plot(t, up_stream, label="up_stream", lw=1, linestyle="--")
        axes[0].set_ylabel("UP_COUNT")
        axes[0].legend()

        axes[1].plot(t, down_full, label="down_full", lw=1)
        axes[1].plot(t, down_stream, label="down_stream", lw=1, linestyle="--")
        axes[1].set_ylabel("DOWN_COUNT")
        axes[1].legend()

        axes[2].plot(t, dt_full, label="Dt_full", lw=1)
        axes[2].plot(t, dt_stream, label="Dt_stream", lw=1, linestyle="--")
        axes[2].set_ylabel("Dt")
        axes[2].legend()

        axes[3].plot(t, sil_full, label="sil_full", lw=1)
        axes[3].plot(t, sil_stream, label="sil_stream", lw=1, linestyle="--")
        axes[3].set_ylabel("SILENCE")
        axes[3].legend()

        axes[4].plot(t, np.abs(up_full - up_stream), label="|up diff|", lw=1)
        axes[4].plot(t, np.abs(down_full - down_stream), label="|down diff|", lw=1)
        axes[4].plot(t, np.abs(dt_full - dt_stream), label="|Dt diff|", lw=1)
        axes[4].set_ylabel("abs diff")
        axes[4].set_xlabel("time [s]")
        axes[4].legend()

        out_path = os.path.join(args.outdir, f"window_streamer_equiv_batch{i:03d}.png")
        fig.tight_layout()
        if args.plot:
            fig.savefig(out_path, dpi=160)
            print("wrote", out_path)
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
