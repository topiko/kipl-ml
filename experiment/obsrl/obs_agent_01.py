import os

import dotenv
import hydra
import mlflow
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from experiment.obsrl.sim import sim
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import Datasets, assets
from kipl_ml.data.wf_dataset import dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import AGENT1
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")


N_SPLITS = 5
TEST_XV = 0
TARGET = assets.PAGE_LABEL
DATASET = Datasets.BIGENOUGH


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="battle-config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "obsrl"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    discriminator_model_id = "m-f87f760721794c8090e741cf3e2456b4"

    model_uri = mlflow.get_logged_model(discriminator_model_id).model_uri

    discriminator = mlflow.pytorch.load_model(model_uri, map_location="cpu")

    feature_names = [Feats.DIRS, Feats.TIMES]
    npackets = 1000

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=npackets),
        **defence_builder.get_defence(cfg),
    )

    dl_train = dl_(ds_train, bs=16, collate_fn=None, shuffle=True)
    dl_valid = dl_(ds_valid, bs=64, collate_fn=None, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs = AGENT1().to(device)

    e = 0
    while True:
        with tqdm(
            dl_train,
            desc=f"epoch {e:02d}",
            ncols=2 * TQDM_W,
        ) as pbar:
            for X, y in pbar:
                X = dict_to_device(X, device)
                y = y.to(device)

                sim(obs, discriminator, X, y)


if __name__ == "__main__":
    main()
