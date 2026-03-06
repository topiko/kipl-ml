my-df-multi

Self-contained df-multi runner modeled after maybenot-gen/scripts/df.py,
but using the original Laserbeak DFNet + feature pipeline (df-multi).

You need to build your env first, see. pyproject.toml for deps.

Main entrypoint:
  df-multi.py

Train + save model:
  python df-multi.py -d /path/to/dataset --train --sm /tmp/dfmulti.pt

Usage example (evaluate only):
  python df-multi.py -d /path/to/dataset --lm /tmp/dfmulti.pt --batchsize 64


Ephemeral defences default (df-multi) equivalent:
  # Matches experiment/ephemeral_defences/config/df-multi.yaml defaults:
  # - dataset: bigenough
  # - model: df-multi (trace_len 10_000)
  # - lr_scheduler: cosine (warmup 10)
  # - optimizer: adamw (lr 0.002, wd 0.001)
  # - loss: crossentropy (label_smoothing 0.1)
  # - train: batch_size 64, patience 15, n_epochs 30
  python df-multi.py -d /PATH/TO/bigenough --train

Notes:
- Defaults to --input-size 10000 to match kipl-ml df-multi configs; pass
  --input-size 7000 to match Laserbeak df-multi.json.
- Feature list defaults to Laserbeak df-multi feature_list.
- Input logs are expected to have lines like: "timestamp,dir,size" where dir
  contains 's' or 'r'. Use --time-unit if timestamps are not nanoseconds.
- By default the script computes features lazily in the PyTorch DataLoader.
  If you want the df.py-style behavior (preload everything into RAM), use
  --preload.
- You can persist preprocessed features with --cache-dir to avoid recomputing
  them across runs.
- Default split is K-bucket cross-validation on sample/subpage index:
  - test bucket: `-f % --xv-splits`
  - valid bucket: `(test-1) % --xv-splits`
  - all other buckets: training
  Defaults: `--xv-splits 10`, `-f 0`.
- During training, the script restores the best (lowest validation-loss) model
  before saving and before final test metrics.
