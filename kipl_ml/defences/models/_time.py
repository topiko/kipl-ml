from __future__ import annotations


def resolve_time_step_s(
    time_step_s: float | None,
    time_step: float | None,
    *,
    default: float | None = None,
) -> float:
    if time_step_s is None:
        if time_step is None:
            if default is None:
                raise TypeError("time_step_s is required")
            return float(default)
        return float(time_step)

    if time_step is not None and float(time_step_s) != float(time_step):
        raise ValueError("time_step and time_step_s must match")
    return float(time_step_s)


def get_time_step_s_attr(instance: object) -> float:
    value = instance.__dict__.get("_time_step_s")
    if value is None:
        value = instance.__dict__["time_step"]
    return float(value)


def set_time_step_s_attr(instance: object, value: float) -> None:
    value = float(value)
    instance.__dict__["_time_step_s"] = value
    instance.__dict__["time_step"] = value
