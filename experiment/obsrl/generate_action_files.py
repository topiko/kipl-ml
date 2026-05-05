import os

import mlflow
import numpy as np
import pandas as pd
import torch

from kipl_ml.data.utils import Datasets, assets, load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.rl.enums import Actions, StepAction, StepActions
from kipl_ml.rl.simulate import policy_rollout
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


def _actions_to_frame(actions: StepActions) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for sa in actions:
        row = {"act_time_bin": int(sa.time)}
        row["do_nothing"] = int(Actions.DO_NOTHING in sa)
        row["send_up_count"] = (
            int(sa[Actions.SEND_UP].count) if Actions.SEND_UP in sa else 0
        )
        row["send_down_count"] = (
            int(sa[Actions.SEND_DOWN].count) if Actions.SEND_DOWN in sa else 0
        )
        row["send_up_after_steps"] = (
            int(sa[Actions.SEND_UP].after_steps) if Actions.SEND_UP in sa else 0
        )
        row["send_down_after_steps"] = (
            int(sa[Actions.SEND_DOWN].after_steps) if Actions.SEND_DOWN in sa else 0
        )
        row["delay_up_steps"] = (
            int(sa[Actions.DELAY_UP].steps) if Actions.DELAY_UP in sa else 0
        )
        row["delay_down_steps"] = (
            int(sa[Actions.DELAY_DOWN].steps) if Actions.DELAY_DOWN in sa else 0
        )
        row["selector"] = (
            int(sa[Actions.SELECTOR].selected) if Actions.SELECTOR in sa else 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


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

    print(meta_ser)

    trace_id = meta_ser.trace_id

    data_dir = DATA_DIR + f"{model_id}/trace_{trace_id}/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(N_REALIZATIONS):
        with torch.no_grad():
            fd, act_times, actions, log_ps, sel_probs, _, entropies, X_obs = (
                policy_rollout(obs, X, sample=True)
            )

        act_times = act_times.squeeze(0).cpu().numpy()
        action_df = _actions_to_frame(actions[0])
        action_df["act_times [s]"] = act_times

        obs_df = pd.DataFrame(
            data={k: v.squeeze(0).cpu().numpy() for k, v in X_obs.items()}
        ).astype({Feats.DIRS: int, Feats.DECOY: int})

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
