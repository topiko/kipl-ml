import os
from functools import partial

import dotenv
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from experiment.trace_gan.data_utils import collate_fn_, dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import assets
from kipl_ml.data.wf_dataset import dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.logging.utils import log_dict
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.trgen import ANTINCLF1
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
from kipl_ml.tools.plottr import plot_bursts
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="battle-config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "rnn-fight"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    discriminator_model_id = "m-aab63f997ef74e7dbe7e61f7d33f7c60"

    model_uri = mlflow.get_logged_model(discriminator_model_id).model_uri

    discriminator = mlflow.pytorch.load_model(model_uri)

    dataset = cfg.dataset.name

    feature_names = [Feats.BURST_LENS, Feats.BURST_RELDURS]

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=None),
        **defence_builder.get_defence(cfg),
    )

    collate_fn = partial(collate_fn_, seq_len=cfg.n_bursts)
    dl_train = dl_(ds_train, bs=cfg.batch_size, collate_fn=collate_fn, shuffle=True)
    dl_valid = dl_(ds_valid, bs=128, collate_fn=collate_fn, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = torch.nn.CrossEntropyLoss()

    obs = ANTINCLF1(feature_names).to(device)

    optimG = torch.optim.Adam(obs.parameters(), lr=0.002)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimG, factor=0.5, patience=3
    )
    e = 0
    with mlflow.start_run():
        while True:
            loss_ = 0
            n = 1
            discriminator.train()
            obs.train()
            with tqdm(dl_train, desc=f"epoch {e:02d}", ncols=TQDM_W) as pbar:
                for X, y in pbar:
                    optimG.zero_grad()
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    X = obs(X)

                    logits, _ = discriminator(X)
                    loss = -loss_fn(
                        logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, cfg.n_bursts)
                    )

                    loss.backward()

                    optimG.step()
                    loss_ += (loss.item() - loss_) / n

                    pbar.set_postfix({"l": loss_, "lr": lr_scheduler.get_last_lr()[0]})

                    n += 1

            e += 1
            lr_scheduler.step(loss_)
            valid_metrics_d = evaluate_model(
                discriminator, dl_valid, [Accuracy()], None, key="valid"
            )
            valid_obs_metrics_d = evaluate_model(
                discriminator,
                dl_valid,
                [Accuracy()],
                None,
                key="valid-obs",
                obsfuscator=obs,
            )

            log_dict(valid_metrics_d)
            log_dict(valid_obs_metrics_d)

            mlflow.log_metrics(valid_metrics_d, step=e)
            mlflow.log_metrics(valid_obs_metrics_d, step=e)

            rng = np.random.default_rng(seed=42)

            ntraces = 2

            idxs = rng.integers(0, len(ds_valid), size=ntraces)

            fig, axarr = plt.subplots(
                2, ntraces, figsize=(ntraces * 7, 7), sharex="col"
            )

            for i, didx in enumerate(idxs):
                X, y = collate_fn([ds_valid[didx]])

                X = dict_to_device(X, device)
                y = y.to(device)

                logits, _ = discriminator(X)

                ax = plot_bursts(
                    X,
                    ax=axarr[0, i],
                    cl_probs=nn.functional.softmax(logits, dim=-1),
                    true_class=y.item(),
                )
                ax.set_title(f"True class: {y.item()}")

                X = obs(X)

                logits_obs = discriminator(X)[0]
                plot_bursts(
                    X,
                    ax=axarr[1, i],
                    cl_probs=nn.functional.softmax(logits_obs, dim=-1),
                )

            fig.canvas.draw()

            if (e - 1) % 10 == 0:
                mlflow.log_figure(fig, f"bursts_clf_epoch={e:03d}.png")

            plt.close()


if __name__ == "__main__":
    main()
