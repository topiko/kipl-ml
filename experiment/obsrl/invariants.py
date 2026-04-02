from __future__ import annotations

from dataclasses import dataclass

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import Actions, NoAction
from kipl_ml.rl.observation import WindowFeatureStreamer
from kipl_ml.trace.enums import Feats
from kipl_ml.utils.time import _time_to_bin_idx


@dataclass
class Row24Report:
    ok: bool
    n_windows: int
    n_nonpadding_packets: int
    total_fd_up: int
    total_fd_down: int
    total_x_up: int
    total_x_down: int
    duplicate_bins_total: int
    per_window_mismatches_total: int
    per_bin_mismatches_total: int
    missing_packet_bins_total: int
    extra_nonzero_fd_bins_total: int
    duplicate_bins_sample: list[tuple[int, int, int]]
    per_window_mismatches_sample: list[tuple[int, int, float, int, int, int, int]]
    per_bin_mismatches_sample: list[tuple[int, int, int, int, int]]
    missing_packet_bins_sample: list[int]
    extra_nonzero_fd_bins_sample: list[int]


@dataclass
class RecomputedFdReport:
    ok: bool
    n_fd: int
    n_ref: int
    mismatch_total: int
    mismatch_sample: list[tuple[int, int, int, int, int, int, int]]


@dataclass
class DelayLeakReport:
    n_delay_steps: int
    bad_all: int
    bad_nonpadding: int
    bad_windows_sample: list[tuple[int, int, int, int, int]]


def _get_fd_bins_and_counts(
    fd: dict[Feats, torch.Tensor], idx: int, dt_s: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w_t = fd[Feats.TIME_BINS][idx]
    # fd TIME_BINS are int bins; -1 = invalid sentinel.
    m_w = w_t >= 0
    w_bins = w_t[m_w].to(torch.long)

    up_fd = torch.round(fd[Feats.UP_COUNT][idx][m_w]).to(torch.long)
    down_fd = torch.round(fd[Feats.DOWN_COUNT][idx][m_w]).to(torch.long)
    return w_t[m_w], w_bins, up_fd, down_fd


def _get_nonpadding_packet_bins_and_dirs(
    X_obs: dict[Feats, torch.Tensor], idx: int, dt_s: float
) -> tuple[torch.Tensor, torch.Tensor]:
    pkt_t = X_obs[Feats.TIMES][idx]
    pkt_d = X_obs[Feats.DIRS][idx]
    pkt_p = X_obs[Feats.DECOY][idx] != 0
    # X_obs times are float seconds; filter non-padding active packets.
    m = (pkt_d != 0) & (~pkt_p)
    pkt_t = pkt_t[m]
    pkt_d = pkt_d[m]
    pkt_bins = _time_to_bin_idx(pkt_t, dt_s)
    return pkt_bins, pkt_d


def _count_by_bin(
    bins: torch.Tensor, dirs: torch.Tensor
) -> tuple[dict[int, int], dict[int, int]]:
    up: dict[int, int] = {}
    down: dict[int, int] = {}
    for b, d in zip(bins.tolist(), dirs.tolist()):
        bb = int(b)
        dd = int(round(float(d)))
        if dd == int(UPLOAD):
            up[bb] = up.get(bb, 0) + 1
        elif dd == int(DOWNLOAD):
            down[bb] = down.get(bb, 0) + 1
    return up, down


def check_row2_equals_row4_minus_padding(
    fd: dict[Feats, torch.Tensor],
    X_obs: dict[Feats, torch.Tensor],
    dt_s: float,
    *,
    idx: int = 0,
    max_report: int = 25,
) -> Row24Report:
    w_t, w_bins, up_fd, down_fd = _get_fd_bins_and_counts(fd, idx=idx, dt_s=dt_s)
    pkt_bins, pkt_dirs = _get_nonpadding_packet_bins_and_dirs(X_obs, idx=idx, dt_s=dt_s)

    w_bins_l = [int(v) for v in w_bins.tolist()]
    up_fd_l = [int(v) for v in up_fd.tolist()]
    down_fd_l = [int(v) for v in down_fd.tolist()]

    x_up, x_down = _count_by_bin(pkt_bins, pkt_dirs)

    seen_bin_first: dict[int, int] = {}
    duplicate_bins_total = 0
    duplicate_bins_sample: list[tuple[int, int, int]] = []
    for i, b in enumerate(w_bins_l):
        if b in seen_bin_first:
            duplicate_bins_total += 1
            if len(duplicate_bins_sample) < max_report:
                duplicate_bins_sample.append((b, seen_bin_first[b], i))
        else:
            seen_bin_first[b] = i

    per_window_mismatches_total = 0
    per_window_mismatches_sample: list[tuple[int, int, float, int, int, int, int]] = []
    for i, b in enumerate(w_bins_l):
        up_x = x_up.get(b, 0)
        down_x = x_down.get(b, 0)
        if up_fd_l[i] != up_x or down_fd_l[i] != down_x:
            per_window_mismatches_total += 1
            if len(per_window_mismatches_sample) < max_report:
                per_window_mismatches_sample.append(
                    (
                        i,
                        b,
                        float(w_t[i].item()),
                        up_fd_l[i],
                        down_fd_l[i],
                        up_x,
                        down_x,
                    )
                )

    fd_up_sum: dict[int, int] = {}
    fd_down_sum: dict[int, int] = {}
    for b, u, d in zip(w_bins_l, up_fd_l, down_fd_l):
        fd_up_sum[b] = fd_up_sum.get(b, 0) + int(u)
        fd_down_sum[b] = fd_down_sum.get(b, 0) + int(d)

    fd_bins_set = set(fd_up_sum.keys()) | set(fd_down_sum.keys())
    x_bins_set = set(x_up.keys()) | set(x_down.keys())

    per_bin_mismatches_total = 0
    per_bin_mismatches_sample: list[tuple[int, int, int, int, int]] = []
    for b in sorted(fd_bins_set | x_bins_set):
        fu = fd_up_sum.get(b, 0)
        fdn = fd_down_sum.get(b, 0)
        xu = x_up.get(b, 0)
        xdn = x_down.get(b, 0)
        if fu != xu or fdn != xdn:
            per_bin_mismatches_total += 1
            if len(per_bin_mismatches_sample) < max_report:
                per_bin_mismatches_sample.append((b, fu, fdn, xu, xdn))

    missing_packet_bins = sorted(x_bins_set - fd_bins_set)
    missing_packet_bins_total = len(missing_packet_bins)
    missing_packet_bins_sample = missing_packet_bins[:max_report]

    extra_nonzero_fd_bins = sorted(
        b
        for b in (fd_bins_set - x_bins_set)
        if fd_up_sum.get(b, 0) != 0 or fd_down_sum.get(b, 0) != 0
    )
    extra_nonzero_fd_bins_total = len(extra_nonzero_fd_bins)
    extra_nonzero_fd_bins_sample = extra_nonzero_fd_bins[:max_report]

    total_fd_up = int(sum(up_fd_l))
    total_fd_down = int(sum(down_fd_l))
    total_x_up = int(sum(x_up.values()))
    total_x_down = int(sum(x_down.values()))

    ok = (
        total_fd_up == total_x_up
        and total_fd_down == total_x_down
        and duplicate_bins_total == 0
        and per_window_mismatches_total == 0
        and per_bin_mismatches_total == 0
        and missing_packet_bins_total == 0
        and extra_nonzero_fd_bins_total == 0
    )

    return Row24Report(
        ok=ok,
        n_windows=len(w_bins_l),
        n_nonpadding_packets=int(pkt_bins.numel()),
        total_fd_up=total_fd_up,
        total_fd_down=total_fd_down,
        total_x_up=total_x_up,
        total_x_down=total_x_down,
        duplicate_bins_total=duplicate_bins_total,
        per_window_mismatches_total=per_window_mismatches_total,
        per_bin_mismatches_total=per_bin_mismatches_total,
        missing_packet_bins_total=missing_packet_bins_total,
        extra_nonzero_fd_bins_total=extra_nonzero_fd_bins_total,
        duplicate_bins_sample=duplicate_bins_sample,
        per_window_mismatches_sample=per_window_mismatches_sample,
        per_bin_mismatches_sample=per_bin_mismatches_sample,
        missing_packet_bins_sample=missing_packet_bins_sample,
        extra_nonzero_fd_bins_sample=extra_nonzero_fd_bins_sample,
    )


def check_fd_matches_recomputed_nonpadding(
    fd: dict[Feats, torch.Tensor],
    X_obs: dict[Feats, torch.Tensor],
    dt_s: float,
    max_silence_s: float,
    *,
    idx: int = 0,
    max_report: int = 25,
) -> RecomputedFdReport:
    X_np = {
        Feats.TIMES: X_obs[Feats.TIMES].clone(),
        Feats.DIRS: X_obs[Feats.DIRS].clone(),
        Feats.DECOY: torch.zeros_like(X_obs[Feats.DECOY]),
    }
    X_np[Feats.DIRS][X_obs[Feats.DECOY] != 0] = 0

    ref_features = [Feats.TIME_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT, Feats.Dt_BINS]
    ref_streamer = WindowFeatureStreamer(X_np, dt_s, max_silence_s, ref_features)
    ref_steps = {f: [] for f in ref_features}
    ref_actions = [NoAction(time=0) for _ in range(X_np[Feats.TIMES].shape[0])]
    for _ in range(10000):
        fd_t, _, active = ref_streamer.step(ref_actions)
        for f in ref_features:
            ref_steps[f].append(fd_t[f])
        if active.sum() == 0:
            break
    fd_ref = {f: torch.cat(ref_steps[f], dim=1) for f in ref_features}

    w_t_a, w_b_a, up_a, down_a = _get_fd_bins_and_counts(fd, idx=idx, dt_s=dt_s)
    w_t_b, w_b_b, up_b, down_b = _get_fd_bins_and_counts(fd_ref, idx=idx, dt_s=dt_s)

    n_a = int(w_b_a.numel())
    n_b = int(w_b_b.numel())
    n = min(n_a, n_b)

    mismatch_total = 0
    mismatch_sample: list[tuple[int, int, int, int, int, int, int]] = []
    for i in range(n):
        ba = int(w_b_a[i].item())
        bb = int(w_b_b[i].item())
        ua = int(up_a[i].item())
        ub = int(up_b[i].item())
        da = int(down_a[i].item())
        db = int(down_b[i].item())
        if ba != bb or ua != ub or da != db:
            mismatch_total += 1
            if len(mismatch_sample) < max_report:
                mismatch_sample.append((i, ba, bb, ua, ub, da, db))

    if n_a != n_b:
        mismatch_total += abs(n_a - n_b)

    ok = mismatch_total == 0
    return RecomputedFdReport(
        ok=ok,
        n_fd=n_a,
        n_ref=n_b,
        mismatch_total=mismatch_total,
        mismatch_sample=mismatch_sample,
    )


def count_packets_inside_delay_windows(
    act_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    X_obs: dict[Feats, torch.Tensor],
    dt_s: float,
    *,
    idx: int = 0,
    max_report: int = 25,
) -> DelayLeakReport:
    if Actions.DELAY_BINS not in actions:
        return DelayLeakReport(
            n_delay_steps=0, bad_all=0, bad_nonpadding=0, bad_windows_sample=[]
        )

    t_act = act_times[idx]
    d_act = actions[Actions.DELAY_BINS][idx]
    # act_times and DELAY are int bins; -1 = invalid.
    m = (t_act >= 0) & (d_act > 0)
    if not bool(m.any().item()):
        return DelayLeakReport(
            n_delay_steps=0, bad_all=0, bad_nonpadding=0, bad_windows_sample=[]
        )

    # Already int bins - no conversion needed.
    delay_bins = d_act[m].to(torch.long)
    t0_bins = t_act[m].to(torch.long)

    pkt_t = X_obs[Feats.TIMES][idx]
    pkt_d = X_obs[Feats.DIRS][idx]
    pkt_p = X_obs[Feats.DECOY][idx] != 0
    # X_obs times are float seconds.
    m_all = pkt_d != 0
    m_np = m_all & (~pkt_p)
    pkt_bins_all = _time_to_bin_idx(pkt_t[m_all], dt_s)
    pkt_bins_np = _time_to_bin_idx(pkt_t[m_np], dt_s)

    bad_all = 0
    bad_nonpadding = 0
    bad_windows_sample: list[tuple[int, int, int, int, int]] = []
    for j, (sb, sh) in enumerate(zip(t0_bins.tolist(), delay_bins.tolist())):
        s = int(sb)
        shift = int(sh)
        if shift <= 0:
            continue
        e = s + shift
        c_all = int(((pkt_bins_all >= s) & (pkt_bins_all < e)).sum().item())
        c_np = int(((pkt_bins_np >= s) & (pkt_bins_np < e)).sum().item())
        bad_all += c_all
        bad_nonpadding += c_np
        if (c_all > 0 or c_np > 0) and len(bad_windows_sample) < max_report:
            bad_windows_sample.append((j, s, e, c_all, c_np))

    return DelayLeakReport(
        n_delay_steps=int(t0_bins.numel()),
        bad_all=bad_all,
        bad_nonpadding=bad_nonpadding,
        bad_windows_sample=bad_windows_sample,
    )


def format_row24_report(rep: Row24Report) -> str:
    lines = [
        "row2(row fd) == row4(X_obs)-padding check",
        f"- ok: {rep.ok}",
        f"- windows={rep.n_windows}, nonpadding_packets={rep.n_nonpadding_packets}",
        (
            "- totals fd(up,down)="
            + f"({rep.total_fd_up},{rep.total_fd_down}) "
            + f"X_obs(up,down)=({rep.total_x_up},{rep.total_x_down})"
        ),
        (
            "- issues dup_bins/per_window/per_bin/missing_bins/extra_nonzero_fd="
            + f"{rep.duplicate_bins_total}/{rep.per_window_mismatches_total}/"
            + f"{rep.per_bin_mismatches_total}/{rep.missing_packet_bins_total}/"
            + f"{rep.extra_nonzero_fd_bins_total}"
        ),
    ]
    if rep.duplicate_bins_sample:
        lines.append(f"- duplicate bins sample: {rep.duplicate_bins_sample}")
    if rep.per_window_mismatches_sample:
        lines.append(
            f"- per-window mismatch sample: {rep.per_window_mismatches_sample}"
        )
    if rep.per_bin_mismatches_sample:
        lines.append(f"- per-bin mismatch sample: {rep.per_bin_mismatches_sample}")
    if rep.missing_packet_bins_sample:
        lines.append(f"- missing packet bins sample: {rep.missing_packet_bins_sample}")
    if rep.extra_nonzero_fd_bins_sample:
        lines.append(
            f"- extra nonzero fd bins sample: {rep.extra_nonzero_fd_bins_sample}"
        )
    return "\n".join(lines)


def format_recomputed_fd_report(rep: RecomputedFdReport) -> str:
    lines = [
        "fd vs recomputed(X_obs-padding) check",
        f"- ok: {rep.ok}",
        f"- n_fd={rep.n_fd}, n_ref={rep.n_ref}, mismatches={rep.mismatch_total}",
    ]
    if rep.mismatch_sample:
        lines.append(f"- mismatch sample: {rep.mismatch_sample}")
    return "\n".join(lines)


def format_delay_leak_report(rep: DelayLeakReport) -> str:
    lines = [
        "delay-window leak check",
        f"- delay_steps={rep.n_delay_steps}",
        f"- packets inside delay windows all/nonpadding={rep.bad_all}/{rep.bad_nonpadding}",
    ]
    if rep.bad_windows_sample:
        lines.append(f"- bad windows sample: {rep.bad_windows_sample}")
    return "\n".join(lines)
