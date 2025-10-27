import os
from functools import partial

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data.utils import assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.trgen import RNNCLF1
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def collate_fn_(
    batch: list[tuple[dict[Feats, torch.Tensor], torch.Tensor]], seq_len: int = 300
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
    bs = len(batch)
    features = batch[0][0].keys()
    X = {f: torch.ones((bs, seq_len), dtype=torch.float) for f in features}
    y = torch.zeros((bs,), dtype=torch.long)
    start_idx = torch.randint(0, 5, (bs,))
    for i, (x_, y_) in enumerate(batch):
        sidx = start_idx[i]
        for f in features:
            xtmp = x_[f][sidx : sidx + seq_len + 1]
            if f == Feats.BURST_DIRS:
                xtmp += 1

            if len(xtmp) < seq_len + 1:
                lenx = len(xtmp)
            else:
                lenx = len(xtmp) - 1

            X[f][i, :lenx] = xtmp[:lenx]
        y[i] = y_

    return X, y


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "lstm-clf"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    trace_len = cfg.trace_len
    dataset = cfg.dataset.name

    feature_names = [Feats.BURST_LENS]

    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=None),
    )

    n_bursts = 300

    collate_fn = partial(collate_fn_, seq_len=n_bursts)

    def dl_(ds: WFDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=24,
            collate_fn=collate_fn,
        )

    dl_train = dl_(ds_train)
    dl_valid = dl_(ds_valid)

    clf = RNNCLF1(ds_train.n_classes, feature_names)
    optimG = torch.optim.Adam(clf.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimG, factor=0.8, patience=3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    clf.to(device)

    min_loss = np.inf
    patience = 5
    e = 0
    c = 0

    with mlflow.start_run():
        while True:
            with tqdm(dl_train, desc=f"epoch {e:02d}", ncols=TQDM_W) as pbar:
                loss_ = 0
                n = 1

                clf.train()

                for X, y in pbar:
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    h = None
                    optimG.zero_grad()

                    probs, _ = clf(X, h)

                    loss = loss_fn(
                        probs.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, n_bursts)
                    )

                    loss.backward()
                    optimG.step()

                    loss_ += (loss.item() - loss_) / n

                    pbar.set_postfix({"l": loss_, "lr": lr_scheduler.get_last_lr()[0]})

                    n += 1
                e += 1

                lr_scheduler.step(loss_)

            c += 1

            valid_metrics_d = evaluate_model(clf, dl_valid, [Accuracy()], loss_fn)
            train_metrics_d = evaluate_model(clf, dl_train, [Accuracy()], loss_fn)
            mlflow.log_metrics(
                metrics={f"valid-{k}": v for k, v in valid_metrics_d.items()}, step=e
            )
            mlflow.log_metrics(
                metrics={f"train-{k}": v for k, v in train_metrics_d.items()}, step=e
            )
            loss_ = valid_metrics_d["loss"]

            if loss_ < min_loss:
                logger.info("Improved loss! %.4f -> %.4f", min_loss, loss_)
                min_loss = loss_
                c = 0

            if c > patience:
                break


if __name__ == "__main__":
    main()
