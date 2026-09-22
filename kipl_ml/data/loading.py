"""DataLoader construction with independent worker-local simulation RNG streams."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

# Follow only RNG-owning components and dataset wrappers, never model parameters,
# trace caches, or metadata tables. Keep this list in sync with new RNG owners.
_RNG_CHILDREN = (
    "base", "dataset", "datasets",  # Informative/DEC, Subset, ConcatDataset
    "defence", "defences", "network_context",
    "simul_rng", "machine_rng", "defence_rng",
    "rtt_sampler", "mbps_sampler", "tor_sampler",
)


def _reseed_rngs(owner, seed: np.random.SeedSequence, seen: set[int]) -> None:
    if owner is None or id(owner) in seen:
        return
    seen.add(id(owner))
    if isinstance(owner, (list, tuple)):
        for child, child_seed in zip(owner, seed.spawn(len(owner))):
            _reseed_rngs(child, child_seed, seen)
        return

    # Inspect owned attributes so forwarding wrappers don't reseed the same base
    # twice or accidentally shadow its attributes. Fixed/indexed deck seeds stay
    # unchanged; only mutable NumPy generators are replaced.
    attributes = getattr(owner, "__dict__", {})
    if isinstance(attributes.get("rng"), np.random.Generator):
        owner.rng = np.random.default_rng(seed.spawn(1)[0])
    for name in _RNG_CHILDREN:
        child = attributes.get(name)
        if child is not None:
            _reseed_rngs(child, seed.spawn(1)[0], seen)


def seed_worker(_worker_id: int) -> None:
    """Top-level callback, usable with fork and spawn, reseeded on worker creation.

    PyTorch assigns a new base seed on every iterator creation and offsets it by
    worker ID. Persistent workers instead keep advancing their private streams.
    Reproducibility assumes the same seed, worker count and loader access order.
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Interop settings may already be fixed in a forked worker.

    seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # PyTorch has already seeded its own worker RNG; don't truncate its seed.
    worker = get_worker_info()
    if worker is not None:
        _reseed_rngs(worker.dataset, np.random.SeedSequence(seed), set())


def default_num_workers() -> int:
    """Bound the automatic count by CPUs available to this process and by 24."""
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    return min(24, cpus)


def make_dataloader(
    dataset, batch_size: int, *, num_workers: int | None = None, **kwargs
) -> DataLoader:
    """Use explicit worker counts, or a bounded automatic count when None.

    Workers are nonpersistent by default so parent-side policy/dataset changes
    reach the next iterator. num_workers=0 retains the main-process RNG behavior.
    """
    return DataLoader(
        dataset, batch_size=batch_size,
        num_workers=default_num_workers() if num_workers is None else num_workers,
        worker_init_fn=seed_worker,
        **kwargs,
    )
