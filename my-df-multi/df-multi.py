#!/usr/bin/env python3

import argparse
import csv
import datetime
import math
import multiprocessing as mp
import os
import random
import sys
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


DEFAULT_MODEL_CONFIG = {
    # Mirrors ../deps/Laserbeak-WF-Classifier/configs/df-multi.json
    "input_size": 7000,
    "filter_grow_factor": 2,
    "channel_up_factor": 5.34,
    "depth_wise": False,
    "feature_list": [
        "time_dirs",
        "times_norm",
        "cumul_norm",
        "iat_dirs",
        "inv_iat_log_dirs",
        "running_rates",
    ],
}


# ---- Laserbeak preprocessing (subset) ----


def rate_estimator(iats: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """Vectorized running average flow rate estimator (Laserbeak)."""
    times = torch.cumsum(iats, dim=0)
    sizes_ = torch.cumsum(sizes, dim=0)
    flow_rate = torch.where(times != 0, sizes_ / times, torch.ones_like(times))
    return flow_rate


class DataProcessor:
    """Feature extraction as in Laserbeak for df-multi.

    Expects input x shaped (N, 3) with columns: [times, sizes, dirs].
    Returns features shaped (N, C) stacked in feature_list order.
    """

    def __init__(self, process_options=("dirs",)):
        self.process_options = tuple(process_options) if process_options else ("dirs",)
        self.input_channels = len(self.process_options)
        assert self.input_channels > 0

    def process(self, x: torch.Tensor) -> torch.Tensor:
        # x is (N,3)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected x to have shape (N,3), got {tuple(x.shape)}")

        times = x.T[0]
        sizes = x.T[1]
        dirs = x.T[2]

        # Laserbeak computes upload/download masks but not needed for df-multi.

        feature_dict: dict[str, torch.Tensor] = {
            "times": times,
            "sizes": sizes,
            "dirs": dirs,
        }

        if "time_dirs" in self.process_options:
            feature_dict["time_dirs"] = times * dirs

        if "times_norm" in self.process_options:
            times_norm = times.clone()
            times_norm -= torch.mean(times_norm)
            denom = torch.amax(torch.abs(times_norm))
            if denom == 0:
                times_norm = torch.zeros_like(times_norm)
            else:
                times_norm /= denom
            feature_dict["times_norm"] = times_norm

        # iats needed for iat_dirs, inv_iat_log_dirs, running_rates
        if (
            "iat_dirs" in self.process_options
            or "inv_iat_log_dirs" in self.process_options
            or "running_rates" in self.process_options
            or "cumul_norm" in self.process_options
        ):
            iats = torch.diff(times, prepend=torch.tensor([0.0], device=times.device))
            feature_dict["iats"] = iats

        if "cumul_norm" in self.process_options:
            size_dirs = sizes * dirs
            cumul = torch.cumsum(size_dirs, dim=0)

            cumul_norm = cumul.clone()
            cumul_norm -= torch.mean(cumul_norm)
            denom = torch.amax(torch.abs(cumul_norm))
            if denom == 0:
                cumul_norm = torch.zeros_like(cumul_norm)
            else:
                cumul_norm /= denom
            feature_dict["cumul_norm"] = cumul_norm

        if "iat_dirs" in self.process_options:
            # adjusted iats by +1 to prevent zeros losing directional representation
            iat_dirs = (1.0 + feature_dict["iats"]) * dirs
            feature_dict["iat_dirs"] = iat_dirs

        if "running_rates" in self.process_options:
            running_rates = rate_estimator(feature_dict["iats"], sizes)
            feature_dict["running_rates"] = running_rates

        if "inv_iat_log_dirs" in self.process_options:
            # Laserbeak uses flow_iats (merged up/down iats). For aligned per-packet
            # times/dirs sequences, that equals iats.
            flow_iats = feature_dict["iats"]
            inv_iat_logs = torch.log(
                torch.nan_to_num((1.0 / flow_iats) + 1.0, nan=1e4, posinf=1e4)
            )
            feature_dict["inv_iat_log_dirs"] = inv_iat_logs * dirs

        # Stack in requested order.
        target_size = max(int(t.numel()) for t in feature_dict.values())

        def fix_size(z: torch.Tensor) -> torch.Tensor:
            if z.numel() < target_size:
                z = F.pad(z, (0, target_size - z.numel()))
            elif z.numel() > target_size:
                z = z[:target_size]
            return z

        feature_stack = [fix_size(feature_dict[opt]) for opt in self.process_options]
        features = torch.nan_to_num(torch.stack(feature_stack, dim=-1))
        return features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)


# ---- Laserbeak DFNet (conv-only subset, same behavior for df-multi) ----


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels: int,
        activation: nn.Module,
        depth_wise: bool = False,
        expand_factor: int = 1,
        drop_p: float = 0.0,
        kernel_size: int = 8,
        res_skip: bool = False,
        max_pool: nn.Module | None = None,
    ):
        super().__init__()

        def conv(in_ch: int, out_ch: int) -> nn.Conv1d:
            return nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding="same",
                groups=in_ch if depth_wise else 1,
            )

        self.cv_block = nn.Sequential(
            conv(channels_in, channels * expand_factor),
            nn.BatchNorm1d(channels * expand_factor),
            activation,
            nn.Dropout(p=drop_p),
            conv(channels * expand_factor, channels),
            nn.BatchNorm1d(channels),
            activation,
        )

        self.use_residual = res_skip
        self.max_pool = max_pool
        if self.use_residual:
            if max_pool is not None:
                stride = getattr(max_pool, "stride", 1)
                if isinstance(stride, tuple):
                    stride = stride[0]
                proj_k = int(stride)
                proj_stride = int(stride)
            else:
                proj_k = 1
                proj_stride = 1
            self.conv_proj = nn.Conv1d(
                channels_in,
                channels,
                kernel_size=proj_k,
                stride=proj_stride,
                padding=proj_k // 2,
                groups=channels_in if depth_wise else 1,
            )
        else:
            self.conv_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r: torch.Tensor | int = 0
        if self.use_residual and self.conv_proj is not None:
            r = self.conv_proj(x)
        x = self.cv_block(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        if not isinstance(r, int):
            r = r[..., : x.size(dim=2)]
            x = r + x
        return x


class DFNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        channel_up_factor: float = 32,
        filter_grow_factor: float = 2,
        stage_count: int = 4,
        input_size: int = 5000,
        depth_wise: bool = False,
        kernel_size: int = 8,
        pool_stride_size: int = 4,
        pool_size: int = 8,
        mlp_hidden_dim: int | list[int] = 512,
        mlp_dropout_p: tuple[float, float] = (0.7, 0.5),
        conv_expand_factor: int = 1,
        block_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.0,
        conv_skip: bool = False,
        use_gelu: bool = False,
        stem_downproj: float = 1.0,
        flatten_feats: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.input_channels = int(input_channels)
        self.kernel_size = int(kernel_size)
        self.pool_stride_size = int(pool_stride_size)
        self.pool_size = int(pool_size)
        self.flatten_feats = bool(flatten_feats)

        self.block_dropout_p = float(block_dropout_p)
        self.conv_dropout_p = float(conv_dropout_p)

        self.filter_grow_factor = float(filter_grow_factor)
        self.conv_expand_factor = int(conv_expand_factor)
        self.conv_skip = bool(conv_skip)
        self.depth_wise = bool(depth_wise)
        self.use_gelu = bool(use_gelu)

        self.stage_count = int(stage_count)
        self.init_filters = int(self.input_channels * float(channel_up_factor))
        self.proj_dim = int(float(stem_downproj) * self.init_filters)
        self.filter_nums = [
            int(self.proj_dim * (self.filter_grow_factor**i))
            for i in range(self.stage_count)
        ]

        self.input_size = int(input_size)
        self.num_classes = int(num_classes)

        if isinstance(mlp_hidden_dim, list):
            mlp_hidden = mlp_hidden_dim
        else:
            mlp_hidden = [int(mlp_hidden_dim)] * 2
        self.mlp_hidden_dim = mlp_hidden

        if isinstance(mlp_dropout_p, tuple) or isinstance(mlp_dropout_p, list):
            mlp_drop = list(mlp_dropout_p)
        else:
            mlp_drop = [float(mlp_dropout_p)] * len(self.mlp_hidden_dim)
        while len(mlp_drop) < len(self.mlp_hidden_dim):
            mlp_drop.append(mlp_drop[-1])
        self.mlp_dropout_p = mlp_drop

        self.stage_sizes = self._stage_size(self.input_size)
        self.fmap_size = self.stage_sizes[-1]

        self._build_model()

    def _build_model(self) -> None:
        self.max_pool = nn.MaxPool1d(
            self.pool_size, stride=self.pool_stride_size, padding=self.pool_size // 2
        )
        self.stage_dropout = nn.Dropout(p=self.block_dropout_p)

        stem_conv = ConvBlock(
            self.input_channels,
            self.init_filters,
            nn.GELU() if self.use_gelu else nn.ELU(),
            depth_wise=self.depth_wise,
            expand_factor=self.conv_expand_factor,
            drop_p=self.conv_dropout_p,
            kernel_size=self.kernel_size,
            res_skip=False,
            max_pool=self.max_pool,
        )
        if self.proj_dim != self.init_filters:
            stem_proj = nn.Conv1d(self.init_filters, self.proj_dim, kernel_size=1)
            stem = nn.Sequential(stem_conv, stem_proj)
        else:
            stem = stem_conv

        self.blocks = nn.ModuleList([stem])

        if self.stage_count > 1:
            for i in range(1, self.stage_count):
                cur_dim = self.filter_nums[i - 1]
                next_dim = self.filter_nums[i]
                conv_block = ConvBlock(
                    cur_dim,
                    next_dim,
                    nn.GELU() if self.use_gelu else nn.ReLU(),
                    depth_wise=False,
                    expand_factor=self.conv_expand_factor,
                    drop_p=self.conv_dropout_p,
                    kernel_size=self.kernel_size,
                    res_skip=self.conv_skip,
                    max_pool=self.max_pool,
                )
                self.blocks.append(nn.ModuleList([conv_block]))

        self.fc_in_features = (
            self.fmap_size * self.filter_nums[-1]
            if self.flatten_feats
            else self.filter_nums[-1] * 2
        )

        fc_layers: list[nn.Module] = [
            nn.Linear(self.fc_in_features, self.mlp_hidden_dim[0]),
            nn.BatchNorm1d(self.mlp_hidden_dim[0]),
            nn.GELU() if self.use_gelu else nn.ReLU(),
            nn.Dropout(self.mlp_dropout_p[0]),
        ]
        for i in range(1, len(self.mlp_hidden_dim)):
            fc_layers.extend(
                [
                    nn.Linear(self.mlp_hidden_dim[i - 1], self.mlp_hidden_dim[i]),
                    nn.BatchNorm1d(self.mlp_hidden_dim[i]),
                    nn.GELU() if self.use_gelu else nn.ReLU(),
                    nn.Dropout(self.mlp_dropout_p[i]),
                ]
            )
        self.fc = nn.Sequential(*fc_layers)
        self.pred = nn.Sequential(nn.Linear(self.mlp_hidden_dim[-1], self.num_classes))

    def _stage_size(self, input_size: int) -> list[int]:
        fmap_size = [int(input_size)]
        for _ in range(len(self.filter_nums)):
            fmap_size.append(
                int(
                    (
                        (fmap_size[-1] - self.pool_size + 2 * (self.pool_size // 2))
                        / self.pool_stride_size
                    )
                    + 1
                )
            )
        return fmap_size[1:]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = cast(nn.Module, self.blocks[0])(x)
        x = self.stage_dropout(x)
        for block in self.blocks[1:]:
            block_ml = cast(nn.ModuleList, block)
            x = cast(nn.Module, block_ml[-1])(x)
            x = self.stage_dropout(x)
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., : self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0, abs(size_dif)))

        x = self.features(x)

        if self.flatten_feats:
            x = x.flatten(start_dim=1)
        else:
            x = torch.cat(
                (
                    torch.max(x, 2).values.flatten(start_dim=1),
                    torch.mean(x, 2).flatten(start_dim=1),
                ),
                dim=1,
            )

        g = self.fc(x)
        y_pred = self.pred(g)
        return y_pred


# ---- Dataset helpers (adapted from maybenot-gen/scripts/df.py) ----


def pad_wg_packet_len(length: int, mtu: int = 1420) -> int:
    return min(mtu, length + (16 - (length % 16)))


def _parse_log_to_raw_tensor(
    log_txt: str,
    time_unit: str,
    use_wg_padding: bool,
    round_time_decimals: int | None,
) -> torch.Tensor:
    """Parse Maybenot .log into (N,3) tensor: [time_s, size, dir]."""

    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []

    def t_to_seconds(t: str) -> float:
        v = float(t)
        if time_unit == "ns":
            v = v / 1e9
        elif time_unit == "us":
            v = v / 1e6
        elif time_unit == "ms":
            v = v / 1e3
        elif time_unit == "s":
            v = v
        else:
            raise ValueError("Invalid time unit")
        if round_time_decimals is not None:
            v = float(f"{v:.{round_time_decimals}f}")
        return v

    for line in log_txt.split("\n"):
        parts = line.split(",")
        if len(parts) < 3:
            break
        d = parts[1]
        if "s" in d:
            dir_ = 1.0
        elif "r" in d:
            dir_ = -1.0
        else:
            continue

        t_s = t_to_seconds(parts[0])
        try:
            sz = int(parts[2])
        except ValueError:
            continue
        sz = abs(sz)
        if use_wg_padding:
            sz = pad_wg_packet_len(sz)

        times.append(t_s)
        sizes.append(float(sz))
        dirs.append(dir_)

    if len(times) == 0:
        return torch.zeros((1, 3), dtype=torch.float32)

    # Ensure times start at 0.
    t0 = times[0]
    times = [t - t0 for t in times]

    x = torch.tensor(np.stack([times, sizes, dirs], axis=1), dtype=torch.float32)
    return x


def _process_one_trace(
    fname: str,
    ID: str,
    feature_list: list[str],
    time_unit: str,
    use_wg_padding: bool,
    round_time_decimals: int | None,
) -> tuple[str, torch.Tensor]:
    # Avoid CPU oversubscription when using multiprocessing.
    try:
        if mp.current_process().name != "MainProcess":
            torch.set_num_threads(1)
    except Exception:
        pass

    with open(fname, "r") as f:
        raw = _parse_log_to_raw_tensor(
            f.read(),
            time_unit=time_unit,
            use_wg_padding=use_wg_padding,
            round_time_decimals=round_time_decimals,
        )

    processor = DataProcessor(feature_list)
    feats = processor(raw)
    return ID, feats


def _process_one_trace_star(args_tuple):
    return _process_one_trace(*args_tuple)


def to_file_label(c: int, sub: int, sample: int) -> str:
    return f"{int(c):04d}-{int(sub):04d}-{int(sample):04d}"


def build_index(
    folder: str,
    classes: int,
    subpages: int | None,
    samples: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Build ID -> filepath and ID -> label mappings."""
    paths: dict[str, str] = {}
    labels: dict[str, int] = {}

    if subpages is None:
        for c in range(0, classes):
            for s in range(0, samples):
                ID = f"{c}-{s}"
                labels[ID] = c
                paths[ID] = os.path.join(folder, str(c), f"{s}.log")
    else:
        for c in range(0, classes):
            for p in range(0, subpages):
                for s in range(0, samples):
                    ID = to_file_label(c, p, s)
                    labels[ID] = c
                    paths[ID] = os.path.join(folder, str(c), f"{ID}.log")

    return paths, labels


def load_dataset(
    folder: str,
    classes: int,
    subpages: int | None,
    samples: int,
    feature_list: list[str],
    workers: int,
    time_unit: str,
    use_wg_padding: bool,
    round_time_decimals: int | None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    labels: dict[str, int] = {}
    todo: list[tuple[str, str, list[str], str, bool, int | None]] = []

    if subpages is None:
        for c in range(0, classes):
            for s in range(0, samples):
                ID = f"{c}-{s}"
                labels[ID] = c
                fname = f"{s}.log"
                todo.append(
                    (
                        os.path.join(folder, str(c), fname),
                        ID,
                        feature_list,
                        time_unit,
                        use_wg_padding,
                        round_time_decimals,
                    )
                )
    else:
        for c in range(0, classes):
            for p in range(0, subpages):
                for s in range(0, samples):
                    ID = to_file_label(c, p, s)
                    labels[ID] = c
                    fname = f"{ID}.log"
                    todo.append(
                        (
                            os.path.join(folder, str(c), fname),
                            ID,
                            feature_list,
                            time_unit,
                            use_wg_padding,
                            round_time_decimals,
                        )
                    )

    data_dict: dict[str, torch.Tensor] = {}

    if workers <= 1:
        for i, item in enumerate(todo, start=1):
            ID, feats = _process_one_trace(*item)
            data_dict[ID] = feats
            if i % 200 == 0 or i == len(todo):
                print(f"{now()} loaded {i}/{len(todo)} traces")
        return data_dict, labels

    # Multiprocessing: use spawn to avoid fork+torch pitfalls.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, maxtasksperchild=200)
    try:
        it = pool.imap_unordered(_process_one_trace_star, todo, chunksize=16)
        for i, (ID, feats) in enumerate(it, start=1):
            data_dict[ID] = feats
            if i % 200 == 0 or i == len(todo):
                print(f"{now()} loaded {i}/{len(todo)} traces")
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        # Terminate quickly and avoid pool finalizer assertion noise.
        pool.terminate()
        pool.join()
        raise
    except Exception:
        pool.terminate()
        pool.join()
        raise

    return data_dict, labels


class Dataset(data.Dataset):
    def __init__(
        self, ids: list[str], dataset: dict[str, torch.Tensor], labels: dict[str, int]
    ):
        self.ids = ids
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ID = self.ids[index]
        return self.dataset[ID], self.labels[ID]


class LogFeatureDataset(data.Dataset):
    """Lazy dataset that reads .log files and computes df-multi features on demand."""

    def __init__(
        self,
        ids: list[str],
        paths: dict[str, str],
        labels: dict[str, int],
        feature_list: list[str],
        time_unit: str,
        use_wg_padding: bool,
        round_time_decimals: int | None,
        cache_dir: str | None = None,
    ):
        self.ids = ids
        self.paths = paths
        self.labels = labels
        self.processor = DataProcessor(feature_list)
        self.time_unit = time_unit
        self.use_wg_padding = use_wg_padding
        self.round_time_decimals = round_time_decimals
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def _cache_path(self, ID: str) -> str:
        assert self.cache_dir is not None
        return os.path.join(self.cache_dir, f"{ID}.pt")

    def __getitem__(self, index):
        ID = self.ids[index]
        y = self.labels[ID]

        if self.cache_dir is not None:
            cp = self._cache_path(ID)
            if os.path.exists(cp):
                feats = torch.load(cp, weights_only=True)
                return feats, y

        with open(self.paths[ID], "r") as f:
            raw = _parse_log_to_raw_tensor(
                f.read(),
                time_unit=self.time_unit,
                use_wg_padding=self.use_wg_padding,
                round_time_decimals=self.round_time_decimals,
            )
        feats = self.processor(raw)

        if self.cache_dir is not None:
            cp = self._cache_path(ID)
            tmp = cp + f".tmp.{os.getpid()}"
            try:
                torch.save(feats, tmp)
                os.replace(tmp, cp)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        return feats, y


def split_dataset_subpages_xv(
    websites: int,
    subpages: int,
    samples: int,
    xv_splits: int,
    test_xv: int,
) -> dict[str, list[str]]:
    """Split IDs into train/valid/test using K cross-validation buckets.

    - Choose `test_xv` as the held-out test bucket.
    - Choose `valid_xv = test_xv - 1` (wrap-around) as validation bucket.
    - All remaining buckets are training.

    Assignment is based on `p % xv_splits` (subpage index modulo K).
    """
    if xv_splits < 3:
        raise ValueError("xv_splits must be >= 3")

    test_idx = int(test_xv) % int(xv_splits)
    valid_idx = (test_idx - 1) % int(xv_splits)

    training: list[str] = []
    validation: list[str] = []
    testing: list[str] = []

    for c in range(0, websites):
        for p in range(0, subpages):
            bucket = int(p) % int(xv_splits)
            for s in range(0, samples):
                ID = to_file_label(c, p, s)
                if bucket == test_idx:
                    testing.append(ID)
                elif bucket == valid_idx:
                    validation.append(ID)
                else:
                    training.append(ID)

    return {"train": training, "valid": validation, "test": testing}


def split_dataset_samples_xv(
    classes: int,
    samples: int,
    xv_splits: int,
    test_xv: int,
) -> dict[str, list[str]]:
    """Split IDs into train/valid/test using K cross-validation buckets.

    Assignment is based on `s % xv_splits` (sample index modulo K).
    """
    if xv_splits < 3:
        raise ValueError("xv_splits must be >= 3")

    test_idx = int(test_xv) % int(xv_splits)
    valid_idx = (test_idx - 1) % int(xv_splits)

    training: list[str] = []
    validation: list[str] = []
    testing: list[str] = []

    for c in range(0, classes):
        for s in range(0, samples):
            ID = f"{c}-{s}"
            bucket = int(s) % int(xv_splits)
            if bucket == test_idx:
                testing.append(ID)
            elif bucket == valid_idx:
                validation.append(ID)
            else:
                training.append(ID)

    return {"train": training, "valid": validation, "test": testing}


def collate_and_pad(batch: list[tuple[torch.Tensor, int]]):
    """Pad variable-length (S,C) features and return (B,C,S)."""
    batch_x, batch_y = zip(*batch)
    batch_y = torch.tensor(batch_y, dtype=torch.long)
    batch_x = torch.nn.utils.rnn.pad_sequence(
        list(batch_x), batch_first=True, padding_value=0.0
    )
    # batch_x: (B, S, C)
    batch_x = batch_x.permute((0, 2, 1))
    return batch_x.float(), batch_y


def get_result(output: torch.Tensor, true_y: torch.Tensor) -> tuple[np.ndarray, float]:
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    accuracy = (pred_y == true_y.numpy()).sum().item() * 1.0 / float(true_y.size(0))
    return pred_y, accuracy


def metrics(threshold: float, predictions: list[list[float]], labels: list[int]):
    tp, fp, fn, accuracy = 0, 0, 0, 0.0
    label_right: dict[int, int] = {}
    label_total: dict[int, int] = {}

    for i in range(len(predictions)):
        label_pred = int(np.argmax(predictions[i]))
        prob_pred = float(max(predictions[i]))
        label_correct = int(labels[i])

        label_total[label_correct] = label_total.get(label_correct, 0) + 1
        label_right[label_correct] = label_right.get(label_correct, 0)

        if prob_pred >= threshold and label_pred == label_correct:
            tp += 1
            label_right[label_pred] = label_right.get(label_pred, 0) + 1
        elif prob_pred >= threshold:
            fp += 1
        else:
            fn += 1

    accuracy = round(float(tp) / float(tp + fp + fn), 4) if (tp + fp + fn) else 0.0
    return tp, fp, fn, accuracy, label_right, label_total


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cosine_warmup_lr_lambda(
    step: int, warmup_steps: int, total_steps: int, num_cycles: float = 0.5
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-d", required=True, default="", help="root folder of client/server dataset"
    )
    ap.add_argument(
        "-c", required=False, type=int, default=95, help="number of classes"
    )
    ap.add_argument(
        "-p", required=False, type=int, default=10, help="number of subpages"
    )
    ap.add_argument(
        "-s", required=False, type=int, default=20, help="samples per subpage to load"
    )
    ap.add_argument(
        "-w", required=False, type=int, default=4, help="workers for loading traces"
    )
    ap.add_argument(
        "-f",
        required=False,
        type=int,
        default=0,
        help="xv fold index for test split (valid is fold-1)",
    )
    ap.add_argument(
        "--xv-splits",
        required=False,
        type=int,
        default=10,
        help="number of cross-validation buckets (K)",
    )

    ap.add_argument(
        "--input-size",
        required=False,
        type=int,
        default=10_000,
        help="model input size (S)",
    )
    ap.add_argument(
        "--time-unit", required=False, choices=["ns", "us", "ms", "s"], default="ns"
    )
    ap.add_argument(
        "--round-time-decimals",
        required=False,
        type=int,
        default=4,
        help="round times to decimals (set -1 to disable)",
    )
    ap.add_argument(
        "--wg-padding",
        required=False,
        default=False,
        action="store_true",
        help="apply WireGuard packet padding (like df.py packetsizes)",
    )

    ap.add_argument(
        "--lm", required=False, default="", help="load model from provided path"
    )
    ap.add_argument(
        "--sm", required=False, default="", help="save model to provided path"
    )
    ap.add_argument(
        "--train",
        required=False,
        default=False,
        action="store_true",
        help="train model",
    )

    ap.add_argument(
        "--preload",
        required=False,
        default=False,
        action="store_true",
        help="preload and preprocess all traces into RAM (df.py style)",
    )
    ap.add_argument(
        "--cache-dir",
        required=False,
        default=None,
        help="optional cache directory for preprocessed feature tensors (lazy mode)",
    )

    ap.add_argument("--epochs", required=False, type=int, default=30, help="epochs")
    ap.add_argument(
        "--patience",
        required=False,
        type=int,
        default=15,
        help="early stopping patience",
    )
    ap.add_argument(
        "--batchsize", required=False, type=int, default=64, help="batch size"
    )
    ap.add_argument("--seed", required=False, type=int, default=42, help="seed")

    ap.add_argument(
        "--optimizer",
        required=False,
        choices=["adamw", "adamax"],
        default="adamw",
        help="optimizer type",
    )
    ap.add_argument(
        "--lr", required=False, type=float, default=0.002, help="learning rate"
    )
    ap.add_argument(
        "--weight-decay", required=False, type=float, default=0.001, help="weight decay"
    )
    ap.add_argument(
        "--scheduler",
        required=False,
        choices=["cosine", "none"],
        default="cosine",
        help="lr scheduler",
    )
    ap.add_argument(
        "--warmup-period",
        required=False,
        type=int,
        default=10,
        help="warmup epochs (cosine scheduler)",
    )
    ap.add_argument("--label-smoothing", required=False, type=float, default=0.1)

    ap.add_argument("--csv", required=False, default=None, help="save metrics as csv")
    ap.add_argument(
        "--extra", required=False, default="", help="extra column value in csv"
    )

    args = vars(ap.parse_args())

    if args["seed"] > -1:
        set_seed(args["seed"])
        print(f"{now()} using deterministic seed {args['seed']}")

    if not os.path.isdir(args["d"]):
        sys.exit(f"{args['d']} is not a directory")

    if args["round_time_decimals"] is not None and int(args["round_time_decimals"]) < 0:
        round_time_decimals = None
    else:
        round_time_decimals = int(args["round_time_decimals"])

    model_cfg = dict(DEFAULT_MODEL_CONFIG)
    model_cfg["input_size"] = int(args["input_size"])  # override
    feature_list = list(
        model_cfg.get("feature_list", DEFAULT_MODEL_CONFIG["feature_list"])
    )

    print(f"{now()} df-multi feature_list={feature_list}")
    print(f"{now()} indexing dataset from {args['d']}")

    subpages = os.path.exists(os.path.join(args["d"], "0", "0000-0000-0000.log"))
    samples = os.path.exists(os.path.join(args["d"], "0", "0.log"))
    if not subpages and not samples:
        sys.exit(f"{args['d']} does not contain subpages or samples")
    if subpages and samples:
        sys.exit(f"{args['d']} contains both subpages and samples")
    print(f"{now()} using subpages" if subpages else f"{now()} using samples")

    paths, labels = build_index(
        args["d"],
        args["c"],
        None if samples else args["p"],
        args["s"],
    )
    print(f"{now()} indexed {len(paths)} items")

    split = (
        split_dataset_samples_xv(args["c"], args["s"], args["xv_splits"], args["f"])
        if samples
        else split_dataset_subpages_xv(
            args["c"], args["p"], args["s"], args["xv_splits"], args["f"]
        )
    )
    print(
        f"{now()} split {len(split['train'])} training, {len(split['valid'])} validation, and {len(split['test'])} testing"
    )

    dataset = None
    if args["preload"]:
        print(f"{now()} preloading and preprocessing dataset (workers={args['w']})")
        dataset, _labels = load_dataset(
            args["d"],
            args["c"],
            None if samples else args["p"],
            args["s"],
            feature_list=feature_list,
            workers=args["w"],
            time_unit=args["time_unit"],
            use_wg_padding=bool(args["wg_padding"]),
            round_time_decimals=round_time_decimals,
        )
        # _labels should match labels
        print(f"{now()} loaded {len(dataset)} items in dataset")

    model = DFNet(
        num_classes=args["c"],
        input_channels=len(feature_list),
        **{k: v for k, v in model_cfg.items() if k != "feature_list"},
    )

    if args["lm"]:
        model = torch.load(args["lm"], weights_only=False)
        print(f"{now()} loaded model from {args['lm']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"{now()} using {torch.cuda.get_device_name(0)}")
    model.to(device)

    if args["train"]:
        print(
            f"{now()} starting training with {args['epochs']} epochs and patience {args['patience']}"
        )
        train_ds: data.Dataset = (
            Dataset(split["train"], dataset, labels)
            if dataset is not None
            else LogFeatureDataset(
                split["train"],
                paths,
                labels,
                feature_list,
                time_unit=args["time_unit"],
                use_wg_padding=bool(args["wg_padding"]),
                round_time_decimals=round_time_decimals,
                cache_dir=args["cache_dir"],
            )
        )

        valid_ids = split["valid"]
        valid_ds: data.Dataset = (
            Dataset(valid_ids, dataset, labels)
            if dataset is not None
            else LogFeatureDataset(
                valid_ids,
                paths,
                labels,
                feature_list,
                time_unit=args["time_unit"],
                use_wg_padding=bool(args["wg_padding"]),
                round_time_decimals=round_time_decimals,
                cache_dir=args["cache_dir"],
            )
        )

        train_gen = data.DataLoader(
            train_ds,
            batch_size=args["batchsize"],
            shuffle=True,
            collate_fn=collate_and_pad,
            num_workers=0 if dataset is not None else max(0, int(args["w"])),
            pin_memory=torch.cuda.is_available(),
        )
        valid_gen = data.DataLoader(
            valid_ds,
            batch_size=args["batchsize"],
            shuffle=False,
            collate_fn=collate_and_pad,
            num_workers=0 if dataset is not None else max(0, int(args["w"])),
            pin_memory=torch.cuda.is_available(),
        )

        if args["optimizer"] == "adamax":
            optimizer = torch.optim.Adamax(params=model.parameters(), lr=args["lr"])
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=args["lr"],
                betas=(0.9, 0.999),
                weight_decay=args["weight_decay"],
            )

            if args["scheduler"] == "cosine":
                total_steps = len(train_gen) * int(args["epochs"])
                warmup_steps = len(train_gen) * int(args["warmup_period"])
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: _cosine_warmup_lr_lambda(
                        step, warmup_steps, total_steps, num_cycles=0.5
                    ),
                )
            else:
                scheduler = None

        criterion = torch.nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=float(args["label_smoothing"])
        )

        best_loss = float("inf")
        best_epoch: int | None = None
        best_state: dict[str, torch.Tensor] | None = None
        patience_left = int(args["patience"])

        for epoch in range(int(args["epochs"])):
            print(f"{now()} epoch {epoch}")

            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            for x, Y in train_gen:
                x = x.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, Y)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                bs = int(Y.size(0))
                train_loss_sum += float(loss.item()) * bs
                train_correct += int((outputs.argmax(dim=1) == Y).sum().item())
                train_total += bs

            train_loss = train_loss_sum / max(1, train_total)
            train_acc = float(train_correct) / float(max(1, train_total))
            print(f"\ttraining loss {train_loss:.4f}")
            print(f"\ttraining accuracy {train_acc:.4f}")
            print(f"\tcur lr {optimizer.param_groups[0]['lr']:.3e}")

            # Simple early stopping on validation loss
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.inference_mode():
                for x, Y in valid_gen:
                    x = x.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)
                    outputs = model(x)
                    loss = criterion(outputs, Y)
                    bs = int(Y.size(0))
                    val_loss_sum += float(loss.item()) * bs
                    val_correct += int((outputs.argmax(dim=1) == Y).sum().item())
                    val_total += bs
            val_loss = val_loss_sum / max(1, val_total)
            val_acc = float(val_correct) / float(max(1, val_total))
            print(f"\tvalidation loss {val_loss:.4f}")
            print(f"\tvalidation accuracy {val_acc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                # Store on CPU to avoid doubling GPU memory.
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_left = int(args["patience"])
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"\tearly stopping, patience {args['patience']} reached")
                    break

        # Restore best model before saving / final test metrics.
        if best_state is not None:
            model.to("cpu")
            model.load_state_dict(best_state, strict=True)
            model.to(device)
            print(
                f"{now()} restored best model from epoch {best_epoch} (val_loss={best_loss:.4f})"
            )

        if args["sm"]:
            torch.save(model, args["sm"])
            print(f"{now()} saved model to {args['sm']}")

    # testing
    test_ds = (
        Dataset(split["test"], dataset, labels)
        if dataset is not None
        else LogFeatureDataset(
            split["test"],
            paths,
            labels,
            feature_list,
            time_unit=args["time_unit"],
            use_wg_padding=bool(args["wg_padding"]),
            round_time_decimals=round_time_decimals,
            cache_dir=args["cache_dir"],
        )
    )
    testing_gen = data.DataLoader(
        test_ds,
        batch_size=args["batchsize"],
        shuffle=False,
        collate_fn=collate_and_pad,
        num_workers=0 if dataset is not None else max(0, int(args["w"])),
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    torch.set_grad_enabled(False)
    predictions: list[list[float]] = []
    p_labels: list[int] = []
    for x, Y in testing_gen:
        x = x.to(device)
        outputs = model(x)
        index = F.softmax(outputs, dim=1).data.cpu().numpy()
        predictions.extend(index.tolist())
        p_labels.extend(Y.data.numpy().tolist())

    print(f"{now()} made {len(predictions)} predictions with {len(p_labels)} labels")

    csvline: list[list[object]] = []
    threshold = np.append([0], 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True))
    threshold = np.around(threshold, decimals=4)
    for th in threshold:
        tp, fp, fn, accuracy, label_right, label_total = metrics(
            th, predictions, p_labels
        )
        print(
            f"\tthreshold {th:4.2}, accuracy {accuracy:4.2}   "
            f"[tp {tp:>5}, fp {fp:>5}, fn {fn:>5}]"
        )
        if th == 0:
            print("\t\t", end="")
            n = 0
            for key, value in sorted(label_right.items(), key=lambda x: x[0]):
                r = value / label_total[key]
                print(f"{key:>2} {r:>5}, ", end=" ")
                n += 1
                if n == 10:
                    print("")
                    print("\t\t", end="")
                    n = 0
            print("")
        csvline.append([th, accuracy, tp, fp, fn, args["extra"]])

    if args["csv"]:
        with open(args["csv"], "w", newline="") as csvfile:
            w = csv.writer(csvfile, delimiter=",")
            w.writerow(["th", "accuracy", "tp", "fp", "fn", "extra"])
            w.writerows(csvline)
        print(f"{now()} saved testing results to {args['csv']}")

    tp, fp, fn, accuracy, _, _ = metrics(0, predictions, p_labels)
    print(accuracy)


if __name__ == "__main__":
    main()
