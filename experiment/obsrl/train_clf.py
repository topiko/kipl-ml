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

from experiment.obsrl.utils import one_batch_train_disc
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import assets
from kipl_ml.data.wf_dataset import dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.logging.utils import log_dict
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
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


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="clf_config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "lstm-clf"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    dataset = cfg.dataset.name
    npackets = cfg.trace_len

    feature_names = [Feats.DIRS, Feats.LOG1P_IATS]

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=npackets),
        **defence_builder.get_defence(cfg),
    )

    dl_train = dl_(
        ds_train, cfg.batch_size, None, shuffle=True, nworkers=None, pin_memory=True
    )
    dl_valid = dl_(ds_valid, 256, None, nworkers=None)

    clf = RNNCLF1(ds_train.n_classes, feature_names, dropout=cfg.dropout)

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
            loss_mean = 0.0
            n = 1
            clf.train()
            with tqdm(dl_train, desc=f"epoch {e:02d}", ncols=TQDM_W) as pbar:
                for X, y in pbar:
                    X = dict_to_device(X, device, non_blocking=True)
                    y = y.to(device)

                    loss, _ = one_batch_train_disc(
                        clf,
                        X,
                        y,
                        optimG,
                        feature_trs=None,
                        train=True,
                        grad_clip=cfg.grad_norm_clip,
                        detach_period=10000,
                        get_accuracy=False,
                    )

                    loss_mean += (loss - loss_mean) / n

                    pbar.set_postfix(
                        {"l": loss_mean, "lr": lr_scheduler.get_last_lr()[0]}
                    )

                    n += 1

            lr_scheduler.step(loss_mean)

            c += 1
            e += 1

            valid_metrics_d = evaluate_model(
                clf, dl_valid, [Accuracy()], loss_fn, key="valid"
            )

            logger.info("Valid metrics:")
            log_dict(valid_metrics_d)
            mlflow.log_metrics(valid_metrics_d, step=e)
            mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=e)

            if (valid_loss := valid_metrics_d["valid-loss"]) < min_loss:
                logger.info("Improved loss! %.4f -> %.4f", min_loss, valid_loss)
                min_loss = valid_loss
                c = 0

            if c > patience:
                break

            if (e - 1) % 10 == 0:
                train_metrics_d = evaluate_model(
                    clf, dl_train, [Accuracy()], loss_fn, key="train"
                )
                logger.info("Train metrics:")
                log_dict(train_metrics_d)

                mlflow.log_metrics(train_metrics_d, step=e)

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

                mlflow.log_figure(fig, f"bursts_clf_epoch={e:03d}.png")

                plt.show()

                plt.close()

        model_info = mlflow.pytorch.log_model(pytorch_model=clf, name="rnnclf")

    model_info = mlflow.get_logged_model(model_info.model_id)

    mlflow.pytorch.load_model(model_info.model_uri)


if __name__ == "__main__":
    main()
