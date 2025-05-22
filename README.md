# KIPL-ml

ML for WF.

### Setup:

Create a virtual env using [uv](https://docs.astral.sh/uv/):

NOTE: you need [rust](https://www.rust-lang.org/tools/install)!

```
uv init
uv sync
```

and then either use `uv run *` or `source .venv/bin/activate` to activate the venv.

To expose a package from a lib use:
`uv init --lib PACKAGE`

E.g. here:
`uv init --lib kipl_ml`

### Data:

Download the "bigenough" dataset from [here](https://dart.cse.kau.se/maybenot/bigenough-95x10x20-standard-rngsubpages.tar.gz).

We rather have a single file for a trace to prepare for potential massive datasets.
In order to make the conversion to this standard use `kipl_ml/data/conversion.py --dataset DATASET`.
There are implementations for "bigenough" and "gong-surakav" datasets.
In order to find these you need to set two paths in `.env` file.

```
.env

WF_DATA_DIR=/path/to/orig/data
MACHINATION=/PATH/TO/MAYBENOT-GEN/target/release/machination

MLFLOW_TRACKING_URI=http://127.0.0.1:8000
MLFLOW_TRACKING_USERNAME=***
MLFLOW_TRACKING_PASSWORD=***
```

### MLFlow:

Run the mlflow server (note the env var from above) - ONLY if you don't have access to external server:

`mlflow server --host 127.0.0.1 --port 8000 --backend-store-uri 'file:///abs/path/to/projectroot/.mlruns' --artifacts-destination '.mlartifacts'`

##### To train the laserbeak models you'll also need:

`git@github.com:topiko/laserbeak.git`
but it is listed as an dep. and should work out of the box.

You also need (DEPRECATED??):

`git@github.com:huggingface/pytorch-image-models.git`

however, those come as depencies and do not require manual installation.

### PyDeps for dep tracking:

Run:

`pydeps path/to/module.py`

to generate a dependency graph of a module. Config file in `.pydeps`.

### Conventions:

From the `bigenough` we currently map the direction flags: "s"(end) and "r"(eceive) into 1 and -1 respectively. We consider dir > 0 as the "upload" direction (according to Laserbeak conventions). See `kipl_ml/data/conversion.py` for the mapping.

### TODO:

Bring attacks form [wf-lib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/).
