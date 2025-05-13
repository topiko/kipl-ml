### Ephemeral-defenses experiments

The structure is as follows: i) we have a `*.sh` for running the experiments. ii) after running the script we run `uv run python *_*.py` to generate tables or figures.

`[dataset]` is always either `bigenough` or `gong-surakav`.

#### 0. Data preparation:

Before one can run anything we need to prepare the data.
Download the datasets from [here](link to dataloading).

Then point in your `.env` the data folder `WF_DATA_DIR` to the folder you downloaded the datasets.
Make sure your dataset folders are named `bigenough`and `gong-surakav`.
Then run the following in the root of this repo!

`python kipl_ml/data/conversion.py --dataset [bigenough/gong-surakav]`

(don't worry about the warnings stating the xv splits are already generated. They are version controlled to ensure same splits for reproducibility.)

This generates some metadata and a metadata dataframe we are using to access the data.

#### 1. Overview tables:

`scripts/overview.sh [dataset]`

Generate the overview tables using:

`python scripts/overview_table.py -en Ephemeral-[dataset]-Aug1-infinite Ephemeral-[dataset]-Aug1-bottleneck`

output is printed in `stdout`, latex formatted table is in `tables/table.tex`.

#### Inftrain overview tables:

`scripts/overview_inftrain.sh [dataset]`

Generate the overview tables using:

`python scripts/overview_table.py -en Ephemeral-[dataset]-inftrain-infinite Ephemeral-[dataset]-inftrain-bottleneck`

output is printed in `stdout`, latex formatted table is in `tables/table.tex`.

#### 2. Cross attack heatmaps:

Here we can leverage the runs from previous steps:

`python scripts/cross_attack_defense_heatmap.py -en Ephemeral-[dataset]-[Aug1/inftrain] --network [infinite/bottleneck/both] --model [df/rf/df-multi/laserbeak]`

output found in `figs/*`

#### 3. Ephemeral defense cost vs. defense level:

`scripts/defense_cost.sh [dataset]`

Figure is then generated using:

`python scripts/defense_cost.py -en Ephemeral-[dataset]-COSTCURVE-infinite`

again the output is found in `figs/*`
