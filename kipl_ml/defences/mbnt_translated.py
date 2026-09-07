from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import mlflow
import numpy as np
import torch
from shared_utils.translate import build_dataset, generate_deck

from kipl_ml.defences.base import DEFENCE_TYPE_KW
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.logging.logger import get_logger
from kipl_ml.network.network import NetworkContext

logger = get_logger(__name__)

_MAYBENOT_DECKS_PATH = os.environ.get("MAYBENOT_DECKS_PATH", ".maybenot-decks")


class MbntTranslated(Maybenot):
    """Maybenot defence whose deck is auto-generated from a trained RNNDefenceAgent.

    The deck is stored at ``$MAYBENOT_DECKS_PATH/<name>_<hash>/`` where
    *hash* is derived from all generation params.  Different params produce
    a different directory so multiple caches coexist.  If the directory
    already exists the deck is reused; otherwise *auto_generate* controls
    whether the deck is built on the fly (blocks until done) or an error
    is raised.

    The ``**maybenot_kwargs`` are forwarded verbatim to
    :class:`Maybenot` — see its constructor for the full list
    (``n_machines``, ``scale``, ``client_padding_budget``, …).
    ``n_machines`` defaults to ``n_traces * n_realizations`` if omitted.
    ``seed`` in *maybenot_kwargs* controls machine dealing (different per
    train/valid/test split); *deck_seed* controls trace selection and the
    ``emit_translate`` RNG (same for all splits).
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        n_traces: int = 100,
        n_realizations: int = 1,
        side: str = "both",
        chaos: bool = False,
        groups: int = 1_000_000_000,
        deck_seed: int = 42,
        network_name: str = "tor",
        tor_profile: int = 1,
        trace_n_packets: int = 5_000,
        trace_trim_beginning: int = 10,
        batch_size: int = 64,
        dataset_name: str = "bigenough",
        auto_generate: bool = True,
        **maybenot_kwargs,
    ):
        self._name = name
        maybenot_kwargs.setdefault("n_machines", n_traces * n_realizations)
        maybenot_kwargs.setdefault("seed", deck_seed)

        deck_path = _resolve_deck(
            name=name,
            model_id=model_id,
            auto_generate=auto_generate,
            n_traces=n_traces,
            n_realizations=n_realizations,
            side=side,
            chaos=chaos,
            groups=groups,
            seed=deck_seed,
            network_name=network_name,
            tor_profile=tor_profile,
            trace_n_packets=trace_n_packets,
            trace_trim_beginning=trace_trim_beginning,
            batch_size=batch_size,
            dataset_name=dataset_name,
        )
        self._deck_dir = deck_path
        super().__init__(
            deck_path=str(deck_path / "machines.txt"),
            **maybenot_kwargs,
        )

    def report(self, to_log: bool = True) -> str:
        str_ = "MbntTranslated Defence:\n"
        str_ += f"\tName: {self._name}\n"
        str_ += f"\tDeck: {self._deck_dir}\n"
        str_ += f"\tN machines: {len(self.machines)}\n"
        str_ += f"\tScale: {self.scale}\n"
        str_ += f"\tFixed per trace: {self.FIXED_PER_TRACE}\n"
        if self.simul_kwargs:
            str_ += "Simul. args\n"
            for k, v in self.simul_kwargs.items():
                str_ += f"\t{k} : {v}\n"
        if to_log:
            from kipl_ml.logging.utils import log_multiline

            log_multiline(str_)
        return str_

    def _mlflow_log_params(self) -> dict[str, str]:
        d = {"name": self._name}
        d["deck"] = str(self._deck_dir)
        d[DEFENCE_TYPE_KW] = "mbnt_translated"
        return d


_PARAMS_FILE = "_gen_params.json"


def _deck_params_hash(model_id: str, **gen_kwargs) -> str:
    params = dict(model_id=model_id, **gen_kwargs)
    raw = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:8]


def _write_gen_params(deck_path: Path, params: dict) -> None:
    (deck_path / _PARAMS_FILE).write_text(json.dumps(params, indent=2, sort_keys=True))


def _resolve_deck(
    *,
    name: str,
    model_id: str,
    auto_generate: bool,
    **gen_kwargs,
) -> Path:
    params_hash = _deck_params_hash(model_id=model_id, **gen_kwargs)
    deck_path = Path(_MAYBENOT_DECKS_PATH) / f"{name}_{params_hash}"

    if deck_path.is_dir():
        logger.info("Reusing cached deck %s", deck_path)
        return deck_path

    if not auto_generate:
        raise RuntimeError(
            f"Deck '{name}' not found at {deck_path}. "
            f"Run: python -m obsrl.translate model_id={model_id} …"
        )
    _generate_deck(deck_path=deck_path, model_id=model_id, **gen_kwargs)
    return deck_path


def _generate_deck(
    *,
    deck_path: Path,
    model_id: str,
    n_traces: int,
    n_realizations: int,
    side: str,
    chaos: bool,
    groups: int,
    seed: int,
    network_name: str,
    tor_profile: int,
    trace_n_packets: int,
    trace_trim_beginning: int,
    batch_size: int,
    dataset_name: str,
) -> None:
    """Generate a deck at *deck_path* (blocking)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s …", model_id)
    obs = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri,
        map_location="cpu",
    ).to(device)

    network_context = NetworkContext(
        network_kind="tor_custom" if network_name.lower() == "tor" else "vpn_custom",
        network_rtt_millis=(5, 20),
        network_mbps=(100_000_000, 100_000_000),
        seed=seed,
        tor_profile=tor_profile if network_name.lower() == "tor" else 0,
    )

    dataset = build_dataset(
        dataset_name,
        network_context,
        seed=seed,
        trace_len=trace_n_packets,
        trim_raw=trace_trim_beginning,
    )

    n_traces_actual = min(n_traces, len(dataset))
    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(len(dataset), size=n_traces_actual, replace=False).tolist()

    cache_root = deck_path.parent
    data_dir = cache_root / f"._data_{deck_path.name}"

    logger.info(
        "Generating deck %s from %d traces × %d realizations …",
        deck_path.name,
        n_traces_actual,
        n_realizations,
    )
    generate_deck(
        obs,
        dataset,
        idxs,
        device=device,
        model_tag=deck_path.name,
        trace_len=trace_n_packets,
        batch_size=batch_size,
        data_dir=str(data_dir),
        output_dir=str(cache_root),
        n_realizations=n_realizations,
        repeat=n_traces_actual * n_realizations,
        seed=seed,
        side=side,
        chaos=chaos,
        groups=groups,
    )

    shutil.rmtree(data_dir, ignore_errors=True)

    _write_gen_params(
        deck_path,
        {
            "model_id": model_id,
            "n_traces": n_traces,
            "n_realizations": n_realizations,
            "side": side,
            "chaos": chaos,
            "groups": groups,
            "seed": seed,
            "network_name": network_name,
            "tor_profile": tor_profile,
            "trace_n_packets": trace_n_packets,
            "trace_trim_beginning": trace_trim_beginning,
            "batch_size": batch_size,
            "dataset_name": dataset_name,
        },
    )
    logger.info("Deck %s ready", deck_path)
