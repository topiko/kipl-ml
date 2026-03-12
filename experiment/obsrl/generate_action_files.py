import os

import mlflow
import numpy as np
import pandas as pd
import torch

from kipl_ml.data.utils import Datasets, assets, load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.rl.action import execute_actions_from_sequence
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.tools.mlflow_utils import set_tracking_uri_from_env
from kipl_ml.trace.features import Feats, FeatureTrs


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

N_SPLITS = 5
TEST_XV = 0
TARGET = assets.PAGE_LABEL
DATASET = Datasets.BIGENOUGH


N_REALIZATIONS = 100

DATA_DIR = "action_data/"
AGENT_IDS = [
    ("m-a9d479ce29e746e696d63724301bec12", 1),
    ("m-61434c1ee703438e9395ede1d83cc596", 10),
    ("m-5a17934e27494d0cbc8c6d60b0bf6151", 20),
    ("m-0805442231744146a8b36250a0dc5067", 30),
    ("m-afee86846e0b443698e237b54e0a134e", 40),
    ("m-848ff31e3c414358b53f891e36e06db4", 50),
]


def generate_for(
    obs,
    model_id: str,
    ds: WFDataset,
    trace_idx: int,
    meta_df: pd.DataFrame,
    device: torch.DeviceObjType,
):
    meta_ser = meta_df.iloc[trace_idx]
    X, y = ds[trace_idx]

    X = dict_to_device(X, device)

    X = {k: x.unsqueeze(0) for k, x in X.items()}

    fd = get_window_feature_dict(
        X, obs.time_step, obs.max_silence_s, features=obs.features
    )

    # We need the seq. lens in forward.
    action_seq_lens = fd.pop(Feats.SEQ_LENS)
    L = action_seq_lens.max().item()

    print(meta_ser)

    trace_id = meta_ser.trace_id

    data_dir = DATA_DIR + f"{model_id}/trace_{trace_id}/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(N_REALIZATIONS):
        with torch.no_grad():
            act_times, actions, log_ps, sel_probs, _, entropies, h = obs.act(
                fd, None, h_detach_period=1000, seq_lens=action_seq_lens
            )

        Xobs = execute_actions_from_sequence(
            X, act_times, actions, time_step_s=float(obs.time_step)
        )
        act_times = act_times.squeeze(0).cpu().numpy()

        action_df = pd.DataFrame(data=act_times, columns=["act_times [s]"])

        for k, v in actions.items():
            action_df[k] = v.squeeze(0).cpu().numpy()

        obs_df = pd.DataFrame(
            data={k: v.squeeze(0).cpu().numpy() for k, v in Xobs.items()}
        ).astype({Feats.DIRS: int, Feats.PADDING: int})

        action_df.to_csv(f"{data_dir}/generated_actions_{i:02d}.csv", index=False)
        obs_df.to_csv(f"{data_dir}/observed_trace_{i:02d}.csv", index=False)


def main():
    feature_names = [Feats.DIRS, Feats.TIMES]

    set_tracking_uri_from_env()

    meta_df = load_dataset_meta_df(DATASET).sort_values("trace_id")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idxs = np.arange(0, len(meta_df), len(meta_df) // 10)
    for agent_id, agent_idx in AGENT_IDS:
        print(agent_idx, agent_idx)
        obs = mlflow.pytorch.load_model(
            mlflow.get_logged_model(agent_id).model_uri,
            map_location="cpu",
        ).to(device)
        ds = WFDataset(
            label=assets.PAGE_LABEL,
            meta_df=meta_df,
            defence=None,
            dataset_key="train",
            feature_trs=FeatureTrs(feature_names=feature_names, n_packets=10_000),
            trim_raw=obs.train_env["trim_beginning"],
        )
        for idx in idxs:
            print(idx)
            generate_for(
                obs,
                f"model_id={agent_idx:03d}",
                ds,
                trace_idx=idx,
                meta_df=meta_df,
                device=device,
            )


if __name__ == "__main__":
    main()
