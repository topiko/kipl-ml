from __future__ import annotations

LegoBrickValue = bool | int | float | str | list[int] | list[float] | list[str]
LegoBrickSpec = dict[str, LegoBrickValue]


def brick_noop() -> LegoBrickSpec:
    return {"kind": "noop"}


def brick_constant_rate_pump(
    *,
    rate_us: float,
    bypass: bool = True,
    replace: bool = True,
) -> LegoBrickSpec:
    return {
        "kind": "ConstantRatePump",
        "rate_us": [rate_us, rate_us],
        "bypass": bypass,
        "replace": replace,
    }


def brick_scheduled_bin(
    *,
    duration_us: float,
    rate_us: float,
    bypass: bool = True,
    replace: bool = True,
) -> LegoBrickSpec:
    return {
        "kind": "ScheduledBin",
        "duration_us": [duration_us, duration_us],
        "rate_us": [rate_us, rate_us],
        "bypass": bypass,
        "replace": replace,
    }


def brick_trailing_cover(
    *,
    rate_us: float,
    limit: int,
    bypass: bool = True,
    replace: bool = True,
) -> LegoBrickSpec:
    return {
        "kind": "TrailingCover",
        "rate_us": [rate_us, rate_us],
        "limit": [limit, limit],
        "bypass": bypass,
        "replace": replace,
    }


def brick_recv_pump(
    *,
    rate_us: float,
    limit: int,
    bypass: bool = True,
    replace: bool = True,
) -> LegoBrickSpec:
    return {
        "kind": "RecvPump",
        "rate_us": [rate_us, rate_us],
        "limit": [limit, limit],
        "bypass": bypass,
        "replace": replace,
    }


def brick_delay_burst(
    *,
    budget: int,
    duration_us: float,
    rate_us: float,
    bypass: bool = True,
    replace: bool = True,
) -> LegoBrickSpec:
    return {
        "kind": "DelayBurst",
        "budget": [budget, budget],
        "duration_us": [duration_us, duration_us],
        "rate_us": [rate_us, rate_us],
        "bypass": bypass,
        "replace": replace,
    }
