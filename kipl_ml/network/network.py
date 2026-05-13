from __future__ import annotations

import random
from collections.abc import Sequence

from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.tools.rng_samplers import NetwkMbps, NetwkRtt

NET_KIND_KW = "network_kind"
NET_DELAY_KW = "network_rtt_millis"
NET_PPS_KW = "network_mbps"
TOR_PROFILE_KW = "tor_profile"
NetworkContextIntDict = dict[str, int]

NETWK_MAP: dict[str, int] = {"vpn_custom": 0, "tor_custom": 1}
INV_NETWK_MAP: dict[int, str] = {v: k for k, v in NETWK_MAP.items()}

# Keys for pre-sampled Tor Metrics values (microseconds, bps)
TOR_E2E_RTT_US_KW = "tor_e2e_rtt_us"
TOR_E2E_TPUT_BPS_KW = "tor_e2e_tput_bps"

TOR_PROFILES: dict[int, dict[str, str]] = {
    0: {},
    1: {
        "tor_metrics_latencies": "2026-02-10,op-de8a,public,20,60,90,130,230",
        "tor_metrics_throughput": "2026-02-10,op-de8a,public,162,8430,14979,24672,46603",
    },
}

# Pre-parsed Tor Metrics box-plot quartiles for Python-side sampling.
# Each entry: (low, q1, md, q3, high)
_TOR_QUARTILES: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    1: (
        (20.0, 60.0, 90.0, 130.0, 230.0),   # latencies (RTT, ms)
        (162.0, 8430.0, 14979.0, 24672.0, 46603.0),  # throughput (kbps)
    ),
}


def _sample_tor_quartile(q: tuple[float, ...]) -> float:
    """Piecewise-uniform sample matching TorMetricsRow::sample() logic.

    Picks one of the four quartile bins uniformly, then samples uniformly
    within that bin.
    """
    bucket = random.randint(0, 3)
    lo, hi = {
        0: (q[0], q[1]),
        1: (q[1], q[2]),
        2: (q[2], q[3]),
        3: (q[3], q[4]),
    }[bucket]
    return random.uniform(lo, hi) if lo != hi else lo


class NetworkContext:
    def __init__(
        self,
        network_kind: str,
        network_rtt_millis: tuple[int, int],
        network_mbps: tuple[int, int],
        seed: int | None,
        tor_profile: int | None,
    ):
        self.network_kind = network_kind
        self.network_rtt_millis = network_rtt_millis
        self.network_mbps = network_mbps
        self.seed = seed
        self.tor_profile = tor_profile

        self.rtt_sampler = NetwkRtt(*network_rtt_millis, seed=seed)
        self.mbps_sampler = NetwkMbps(*network_mbps, seed=seed)

    @classmethod
    def from_cfg(cls, cfg) -> NetworkContext:
        return cls(
            network_kind=cfg.network_kind,
            network_rtt_millis=(int(cfg.rtt_millis.min), int(cfg.rtt_millis.max)),
            network_mbps=(int(cfg.mbps.min), int(cfg.mbps.max)),
            seed=cfg.seed,
            tor_profile=cfg.tor_profile,
        )

    def with_seed(self, seed: int | None) -> NetworkContext:
        return type(self)(
            network_kind=self.network_kind,
            network_rtt_millis=self.network_rtt_millis,
            network_mbps=self.network_mbps,
            seed=seed,
            tor_profile=self.tor_profile,
        )

    def sample_params(self) -> NetworkContextIntDict:
        d: NetworkContextIntDict = {
            NET_KIND_KW: NETWK_MAP[self.network_kind],
            NET_DELAY_KW: int(self.rtt_sampler()),
            NET_PPS_KW: int(self.mbps_sampler()),
            TOR_PROFILE_KW: int(self.tor_profile),
        }

        # Pre-sample Tor Metrics when using a profile with data.
        if (tor_int := self.tor_profile) in _TOR_QUARTILES:
            lat_q, tp_q = _TOR_QUARTILES[tor_int]
            e2e_rtt_ms = _sample_tor_quartile(lat_q)
            e2e_kbps = _sample_tor_quartile(tp_q)
            d[TOR_E2E_RTT_US_KW] = int(e2e_rtt_ms * 1000)   # ms → µs
            d[TOR_E2E_TPUT_BPS_KW] = int(e2e_kbps * 1000)   # kbps → bps

        return d

    @staticmethod
    def to_rust_args(int_dict: NetworkContextIntDict) -> tuple[str, dict[str, object]]:
        network_type = INV_NETWK_MAP[int_dict[NET_KIND_KW]]

        kwargs: dict[str, object] = {
            "rtt_millis": int(int_dict[NET_DELAY_KW]),
            "mbps": int(int_dict[NET_PPS_KW]),
        }

        if (tor_int := int_dict[TOR_PROFILE_KW]) != -1:
            kwargs.update(TOR_PROFILES[tor_int])

        # Forward pre-sampled Tor values (if present) so the Rust side can
        # build the network deterministically — no RNG needed during create().
        if TOR_E2E_RTT_US_KW in int_dict:
            kwargs[TOR_E2E_RTT_US_KW] = int(int_dict[TOR_E2E_RTT_US_KW])
        if TOR_E2E_TPUT_BPS_KW in int_dict:
            kwargs[TOR_E2E_TPUT_BPS_KW] = int(int_dict[TOR_E2E_TPUT_BPS_KW])

        return network_type, kwargs

    @staticmethod
    def to_rust_args_batch(
        batch_dict: dict[str, int | Sequence[int] | object],
    ) -> tuple[str, dict[str, object]]:
        def _as_int_list(value: int | Sequence[int] | object, key: str) -> list[int]:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [int(v) for v in value]
            if hasattr(value, "tolist"):
                out = value.tolist()
                if isinstance(out, list):
                    return [int(v) for v in out]
                return [int(out)]
            return [int(value)]

        kinds = _as_int_list(batch_dict[NET_KIND_KW], NET_KIND_KW)
        if len(set(kinds)) != 1:
            raise ValueError("Batched network_context requires one shared network_kind")
        rtts = _as_int_list(batch_dict[NET_DELAY_KW], NET_DELAY_KW)
        mbps = _as_int_list(batch_dict[NET_PPS_KW], NET_PPS_KW)
        tor_profiles = _as_int_list(batch_dict.get(TOR_PROFILE_KW, -1), TOR_PROFILE_KW)
        if len(set(tor_profiles)) != 1:
            raise ValueError("Batched network_context requires one shared tor_profile")

        network_type = INV_NETWK_MAP[kinds[0]]
        kwargs: dict[str, object] = {
            "rtt_millis": rtts,
            "mbps": mbps,
        }
        if tor_profiles[0] != -1:
            kwargs.update(TOR_PROFILES[tor_profiles[0]])

        # Forward pre-sampled Tor values (batched as lists).
        if TOR_E2E_RTT_US_KW in batch_dict:
            kwargs[TOR_E2E_RTT_US_KW] = _as_int_list(
                batch_dict[TOR_E2E_RTT_US_KW], TOR_E2E_RTT_US_KW
            )
        if TOR_E2E_TPUT_BPS_KW in batch_dict:
            kwargs[TOR_E2E_TPUT_BPS_KW] = _as_int_list(
                batch_dict[TOR_E2E_TPUT_BPS_KW], TOR_E2E_TPUT_BPS_KW
            )

        return network_type, kwargs

    def report(self) -> str:
        lines = key_val_fmt("network_kind", self.network_kind)
        lines += key_val_fmt("rtt", self.rtt_sampler)
        lines += key_val_fmt("mbps", self.mbps_sampler)
        if (tor_int := self.tor_profile) != -1:
            for k, v in TOR_PROFILES[tor_int].items():
                lines += key_val_fmt(k, v)

        return lines
