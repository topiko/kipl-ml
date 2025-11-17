import os

import dotenv
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import assets
from kipl_ml.data.wf_dataset import dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.logging.utils import log_dict
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.models import _WrapPacketProbsNet
from kipl_ml.models.trgen import RNNCLF1
from kipl_ml.models.utils import count_parameters
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
from kipl_ml.tools.plottr import plot_trace
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "lstm-clf"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    dataset = cfg.dataset.name
    npackets = 10000

    feature_names = [Feats.DIRS, Feats.TIMES]

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=npackets),
        **defence_builder.get_defence(cfg),
    )

    dl_train = dl_(ds_train, cfg.batch_size, None, shuffle=True, nworkers=None)
    dl_valid = dl_(ds_valid, 256, None, nworkers=None)

    clf = RNNCLF1(ds_train.n_classes, feature_names, dropout=cfg.dropout)

    # clf = _WrapPacketProbsNet(clf)

    logger.info(f"Model parameters: {count_parameters(clf)}")

    optimG = torch.optim.Adam(clf.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimG, factor=0.5, patience=3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    clf.to(device)

    min_loss = np.inf
    patience = 20
    e = 0
    c = 0

    with mlflow.start_run():
        mlflow.log_params(cfg)
        while True:
            loss_ = 0.0
            n = 1
            clf.train()
            with tqdm(dl_train, desc=f"epoch {e:02d}", ncols=TQDM_W) as pbar:
                for X, y in pbar:
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    h = None
                    optimG.zero_grad()

                    logits, _ = clf(X, h)

                    loss = loss_fn(
                        logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, npackets)
                    )

                    loss.backward()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        clf.parameters(), cfg.grad_norm_clip, error_if_nonfinite=False
                    )

                    optimG.step()

                    loss_ += (loss.item() - loss_) / n

                    pbar.set_postfix({"l": loss_, "lr": lr_scheduler.get_last_lr()[0]})

                    n += 1

            lr_scheduler.step(loss_)

            c += 1
            e += 1

            valid_metrics_d = evaluate_model(
                clf, dl_valid, [Accuracy()], loss_fn, key="valid"
            )
            train_metrics_d = evaluate_model(
                clf, dl_train, [Accuracy()], loss_fn, key="train"
            )

            logger.info("Valid metrics:")
            log_dict(valid_metrics_d)

            logger.info("Train metrics:")
            log_dict(train_metrics_d)

            mlflow.log_metrics(valid_metrics_d, step=e)
            mlflow.log_metrics(train_metrics_d, step=e)
            mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=e)

            if (loss_ := valid_metrics_d["valid-loss"]) < min_loss:
                logger.info("Improved loss! %.4f -> %.4f", min_loss, loss_)
                min_loss = loss_
                c = 0

            if c > patience:
                break

            rng = np.random.default_rng(seed=42)

            ntraces = 5

            idxs = rng.integers(0, len(ds_valid), size=ntraces)

            fig, axarr = plt.subplots(
                ntraces, 1, figsize=(10, ntraces * 3), sharex=True
            )

            for i, ax in zip(idxs, axarr):
                X, y = ds_valid[i]

                X = dict_to_device(X, device)
                y = y.to(device)

                logits, _ = clf({k: v.unsqueeze(0) for k, v in X.items()})

                ax = plot_trace(
                    X,
                    ax=ax,
                    cl_probs=nn.functional.softmax(logits, dim=-1),
                    true_class=y.item(),
                )
                ax.set_title(f"True class: {y.item()}")

            fig.canvas.draw()

            if (e - 1) % 10 == 0:
                mlflow.log_figure(fig, f"bursts_clf_epoch={e:03d}.png")

            plt.show()

            plt.close()

        model_info = mlflow.pytorch.log_model(pytorch_model=clf, name="rnnclf")

    model_info = mlflow.get_logged_model(model_info.model_id)

    mlflow.pytorch.load_model(model_info.model_uri)


if __name__ == "__main__":
    main()
