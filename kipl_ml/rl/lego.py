from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias, cast

LegoBrickValue = bool | int | str | list[str]
LegoBrickSpec = dict[str, LegoBrickValue]
LegoParamScalar: TypeAlias = bool | int | float | str
LegoParamValue: TypeAlias = LegoParamScalar | tuple[LegoParamScalar, ...]
PathLike = str | Path

class LegoBrickSide(StrEnum):
    CLIENT = "client"
    SERVER = "server"
    EITHER = "either"


@dataclass(frozen=True)
class LegoBrick:
    """Semantic selector item; compiled machine strings are a backend artifact."""

    name: str
    kind: str
    side: LegoBrickSide = LegoBrickSide.EITHER
    params: dict[str, LegoParamValue] = field(default_factory=dict)
    machines: tuple[str, ...] = ()
    source: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def noop(cls) -> LegoBrick:
        return cls(name="noop", kind="noop")

    @classmethod
    def compiled_machines(
        cls,
        name: str,
        machines: Sequence[str],
        side: LegoBrickSide,
        source: str | None = None,
        tags: Sequence[str] = (),
    ) -> LegoBrick:
        return cls(
            name=name,
            kind="compiled_machines",
            side=side,
            machines=tuple(machines),
            source=source,
            tags=tuple(tags),
        )

    def to_backend_spec(self) -> LegoBrickSpec:
        if self.kind == "noop":
            return brick_noop()
        if self.machines:
            return brick_compiled_machines(self.machines, name=self.name)
        raise ValueError(f"Lego brick {self.name!r} has no compiled backend")


def brick_noop() -> LegoBrickSpec:
    return {"kind": "noop"}


def brick_compiled_machines(
    machines: Sequence[str], name: str | None = None
) -> LegoBrickSpec:
    spec: LegoBrickSpec = {"kind": "machines", "machines": list(machines)}
    if name is not None:
        spec["name"] = name
    return spec


def to_backend_brick_spec_list(bricks: Sequence[LegoBrick]) -> list[LegoBrickSpec]:
    return [brick.to_backend_spec() for brick in bricks]


def to_backend_brick_specs(
    client_bricks: Sequence[LegoBrick],
    server_bricks: Sequence[LegoBrick],
) -> tuple[list[LegoBrickSpec], list[LegoBrickSpec]]:
    if len(client_bricks) != len(server_bricks):
        raise ValueError("client and server brick lists must have the same length")
    return to_backend_brick_spec_list(client_bricks), to_backend_brick_spec_list(
        server_bricks
    )


def load_lego_brick_specs_from_defense_files(
    paths: Sequence[PathLike], include_noop: bool = True
) -> tuple[list[LegoBrickSpec], list[LegoBrickSpec]]:
    client_bricks, server_bricks = load_lego_bricks_from_defense_files(
        paths,
        include_noop=include_noop,
    )
    return to_backend_brick_specs(client_bricks, server_bricks)


def load_lego_bricks_from_defense_files(
    paths: Sequence[PathLike], include_noop: bool = True
) -> tuple[list[LegoBrick], list[LegoBrick]]:
    """Build aligned brick lists from side-local machines in defense files."""

    client = [LegoBrick.noop()] if include_noop else []
    server = [LegoBrick.noop()] if include_noop else []
    for raw_path in paths:
        path = Path(raw_path)
        name, client_machines, server_machines = _read_defense_file(path)
        source = str(path)
        for idx in range(max(len(client_machines), len(server_machines))):
            if idx < len(client_machines):
                client.append(
                    LegoBrick.compiled_machines(
                        name=f"{name}:c{idx}",
                        machines=[client_machines[idx]],
                        side=LegoBrickSide.CLIENT,
                        source=source,
                        tags=("atlas", "brick"),
                    )
                )
            else:
                client.append(LegoBrick.noop())

            if idx < len(server_machines):
                server.append(
                    LegoBrick.compiled_machines(
                        name=f"{name}:s{idx}",
                        machines=[server_machines[idx]],
                        side=LegoBrickSide.SERVER,
                        source=source,
                        tags=("atlas", "brick"),
                    )
                )
            else:
                server.append(LegoBrick.noop())

    return client, server


def load_lego_static_defense_specs_from_defense_files(
    paths: Sequence[PathLike], include_noop: bool = True
) -> tuple[list[LegoBrickSpec], list[LegoBrickSpec]]:
    client_bricks, server_bricks = load_lego_static_defenses_from_files(
        paths,
        include_noop=include_noop,
    )
    return to_backend_brick_specs(client_bricks, server_bricks)


def load_lego_static_defenses_from_files(
    paths: Sequence[PathLike], include_noop: bool = True
) -> tuple[list[LegoBrick], list[LegoBrick]]:
    """Build selector lists where each item is a complete static defense."""

    client = [LegoBrick.noop()] if include_noop else []
    server = [LegoBrick.noop()] if include_noop else []
    for raw_path in paths:
        path = Path(raw_path)
        name, client_machines, server_machines = _read_defense_file(path)
        source = str(path)
        client.append(
            LegoBrick.compiled_machines(
                name=name,
                machines=client_machines,
                side=LegoBrickSide.CLIENT,
                source=source,
                tags=("atlas", "static-defense"),
            )
        )
        server.append(
            LegoBrick.compiled_machines(
                name=name,
                machines=server_machines,
                side=LegoBrickSide.SERVER,
                source=source,
                tags=("atlas", "static-defense"),
            )
        )
    return client, server


def lego_brick_specs_from_atlas(
    atlas_dir: PathLike,
    include_noop: bool = True,
    include_baselines: bool = False,
) -> tuple[list[LegoBrickSpec], list[LegoBrickSpec]]:
    client_bricks, server_bricks = lego_bricks_from_atlas(
        atlas_dir=atlas_dir,
        include_noop=include_noop,
        include_baselines=include_baselines,
    )
    return to_backend_brick_specs(client_bricks, server_bricks)


def lego_bricks_from_atlas(
    atlas_dir: PathLike,
    include_noop: bool = True,
    include_baselines: bool = False,
) -> tuple[list[LegoBrick], list[LegoBrick]]:
    """Return semantic Lego bricks from current atlas Lego-member machines."""

    paths = atlas_lego_defense_files(
        atlas_dir=atlas_dir,
        include_baselines=include_baselines,
    )
    return load_lego_bricks_from_defense_files(
        paths,
        include_noop=include_noop,
    )


def atlas_lego_defense_files(
    atlas_dir: PathLike,
    include_baselines: bool = False,
) -> list[Path]:
    """Return `defense.def` paths for atlas members in atlas selection order."""

    root = Path(atlas_dir)
    selected = _atlas_member_names(root, include_baselines=include_baselines)
    by_name = _roster_defense_files_by_name(root)

    missing = [name for name in selected if name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"atlas roster is missing defense files for: {', '.join(missing)}"
        )

    return [by_name[name] for name in selected]


def _atlas_member_names(atlas_dir: Path, include_baselines: bool) -> list[str]:
    atlas_json = atlas_dir / "results" / "atlas.json"
    raw: object = json.loads(atlas_json.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{atlas_json} must contain a JSON object")

    members = raw.get("members")
    if not isinstance(members, list):
        raise ValueError(f"{atlas_json} must contain a 'members' list")

    names: list[str] = []
    for idx, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError(f"{atlas_json} members[{idx}] must be an object")
        name = member.get("name")
        baseline = member.get("baseline", False)
        if not isinstance(name, str):
            raise ValueError(f"{atlas_json} members[{idx}]['name'] must be a string")
        if not isinstance(baseline, bool):
            raise ValueError(f"{atlas_json} members[{idx}]['baseline'] must be a bool")
        if include_baselines or not baseline:
            names.append(name)
    return names


def _roster_defense_files_by_name(atlas_dir: Path) -> dict[str, Path]:
    roster_dir = atlas_dir / "roster"
    out: dict[str, Path] = {}
    for path in sorted(roster_dir.glob("*/defense.def"), key=_roster_sort_key):
        name = _origin_name(path.parent / "ORIGIN.md")
        out[name] = path
    return out


def _roster_sort_key(path: Path) -> tuple[int, str]:
    parent = path.parent.name
    if parent.isdecimal():
        return int(parent), parent
    return 2**63 - 1, parent


def _origin_name(path: Path) -> str:
    first_line = path.read_text().splitlines()[0]
    if not first_line.startswith("# "):
        raise ValueError(f"{path} must start with a Markdown title")
    return first_line[2:]


def _read_defense_file(path: Path) -> tuple[str, list[str], list[str]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} must contain a description and a defense JSON line")

    raw: object = json.loads(lines[1])
    if not isinstance(raw, dict):
        raise ValueError(f"{path} defense JSON must be an object")
    defense = cast(dict[str, object], raw)

    return (
        path.parent.name,
        _string_list(defense.get("client"), "client", path),
        _string_list(defense.get("server"), "server", path),
    )


def _string_list(value: object, field: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} field {field!r} must be a list of machine strings")
    return cast(list[str], value)
