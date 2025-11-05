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
from kipl_ml.metrics.overhead_metrics import (
    BurstLenOverhead,
    BurstRelDurOverhead,
)
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.model_eval.obsfuscator import evaluate_obs
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
    dl_valid = dl_(ds_valid, bs=64, collate_fn=collate_fn, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = torch.nn.CrossEntropyLoss()
    dur_loss = BurstRelDurOverhead()
    len_loss = BurstLenOverhead()

    obs = ANTINCLF1(feature_names).to(device)

    optimG = torch.optim.Adam(obs.parameters(), lr=0.001)
    optimD = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    obs_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimG, factor=0.5, patience=3
    )
    disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimD, factor=0.5, patience=3
    )
    e = 0
    with mlflow.start_run():
        optim_obs = True
        while True:
            mean_obsfusc_loss = 0
            mean_len_loss = 0
            mean_dur_loss = 0
            discriminator_loss_ = 0
            n1 = 1
            n2 = 1
            obsfuscator_train_frac = 2
            discriminator.train()
            obs.train()
            logger.info("Obsfuscator train: %s", optim_obs)
            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    optim_obs = True
                    if (n1 / n2) > obsfuscator_train_frac:
                        optim_obs = False
                    optimG.zero_grad()
                    optimD.zero_grad()

                    X = dict_to_device(X, device)
                    y = y.to(device)

                    X_ = obs(X)

                    if not optim_obs:
                        X_ = {k: v.detach() for k, v in X_.items()}
                    logits, _ = discriminator(X_)
                    clf_loss = loss_fn(
                        logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, cfg.n_bursts)
                    )
                    if optim_obs:
                        dur_loss_ = dur_loss(X_, X)
                        len_loss_ = len_loss(X_, X)

                        obsfusc_loss = -clf_loss + dur_loss_ + len_loss_

                        obsfusc_loss.backward()

                        nn.utils.clip_grad_norm_(
                            obs.parameters(),
                            cfg.grad_norm_clip,
                            error_if_nonfinite=False,
                        )

                        optimG.step()
                        mean_obsfusc_loss += (
                            obsfusc_loss.item() - mean_obsfusc_loss
                        ) / n1
                        mean_dur_loss += (dur_loss_.item() - mean_dur_loss) / n1
                        mean_len_loss += (len_loss_.item() - mean_len_loss) / n1
                        n1 += 1
                    else:
                        clf_loss.backward()
                        nn.utils.clip_grad_norm_(
                            discriminator.parameters(),
                            cfg.grad_norm_clip,
                            error_if_nonfinite=False,
                        )
                        optimD.step()
                        discriminator_loss_ += (
                            clf_loss.item() - discriminator_loss_
                        ) / n2
                        n2 += 1

                    pbar.set_postfix(
                        {
                            "lobs": f"{mean_obsfusc_loss:.02f}",
                            "dur_loss": f"{mean_dur_loss:.05f}",
                            "len_loss": f"{mean_len_loss:.03f}",
                            "ldisc": f"{discriminator_loss_:.02f}",
                        }
                    )

            e += 1
            # if optim_obs:
            #     obs_lr_scheduler.step(obsfusc_loss_)
            # else:
            #     disc_lr_scheduler.step(discriminator_loss_)

            valid_metrics_d = evaluate_model(
                discriminator, dl_valid, [Accuracy()], None, key="valid"
            )
            valid_obs_metrics_d = evaluate_model(
                discriminator,
                dl_valid,
                [Accuracy()],
                loss_fn,
                key="valid-obs",
                obsfuscator=obs,
            )

            valid_obs_overhead_metrics_d = evaluate_obs(
                obs, dl_valid, [len_loss, dur_loss], "valid-obs-losses"
            )

            log_dict(valid_metrics_d)
            log_dict(valid_obs_metrics_d)
            log_dict(valid_obs_overhead_metrics_d)
            log_dict(
                {
                    "lro": obs_lr_scheduler.get_last_lr()[0],
                    "lrd": disc_lr_scheduler.get_last_lr()[0],
                }
            )

            optim_obs = discriminator_loss_ < 1.0

            mlflow.log_metrics(valid_metrics_d, step=e)
            mlflow.log_metrics(valid_obs_metrics_d, step=e)
            mlflow.log_metrics(valid_obs_overhead_metrics_d, step=e)

            rng = np.random.default_rng(seed=42)

            ntraces = 2

            idxs = rng.integers(0, len(ds_valid), size=ntraces)

            fig, axarr = plt.subplots(
                2, ntraces, figsize=(ntraces * 7, 7), sharex="col", sharey=True
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

                Xobs = obs(X)

                logits_obs = discriminator(Xobs)[0]
                ax = plot_bursts(
                    Xobs,
                    ax=axarr[1, i],
                    cl_probs=nn.functional.softmax(logits_obs, dim=-1),
                    true_class=y.item(),
                )

                ax.set_title(f"Overhead {len_loss(Xobs, X)}")

            fig.canvas.draw()

            if (e - 1) % 10 == 0:
                mlflow.log_figure(fig, f"bursts_clf_epoch={e:03d}.png")

            plt.close()


if __name__ == "__main__":
    main()
