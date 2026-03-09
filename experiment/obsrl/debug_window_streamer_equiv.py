import numpy as np
import torch

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
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    dt = 0.02
    max_silence_s = 0.2
    extend_end_s = 2.0

    X = _make_synth_batch(bs=8, n_packets=600, dt=dt, device=device)
    features = [Feats.UP_COUNT, Feats.DOWN_COUNT, Feats.Dt, Feats.TIMES, Feats.SILENCE_FLAG]

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

    # Safety bound: worst case we insert one window every dt up to last time + extend.
    max_steps = int((X[Feats.TIMES].max().item() + extend_end_s) / dt) + 10

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
    seq_lens_stream = fd_stream[Feats.TIMES].isfinite().sum(dim=1).long()
    torch.testing.assert_close(seq_lens_stream, seq_lens_full, atol=0, rtol=0)

    # Ensure same max length.
    L_full = fd_full[Feats.TIMES].shape[1]
    L_stream = fd_stream[Feats.TIMES].shape[1]
    if L_stream != L_full:
        raise AssertionError(f"Length mismatch: stream={L_stream} full={L_full}")

    # Exact compare on finite positions.
    mask = fd_full[Feats.TIMES].isfinite()
    for f in features:
        a = fd_full[f]
        b = fd_stream[f]
        atol = 0.0
        if f in {Feats.Dt, Feats.TIMES}:
            atol = 1e-6
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
                print("full.times", fd_full[Feats.TIMES][i0, :10])
                print("stream.times", fd_stream[Feats.TIMES][i0, :10])
                print("full.dt", fd_full[Feats.Dt][i0, :10])
                print("stream.dt", fd_stream[Feats.Dt][i0, :10])
            raise e

        # NaNs in the tail should align too.
        if not torch.equal(a.isnan(), b.isnan()):
            raise AssertionError(f"NaN mask mismatch for {f}")

    print("OK: WindowFeatureStreamer matches get_window_feature_dict")


if __name__ == "__main__":
    main()
