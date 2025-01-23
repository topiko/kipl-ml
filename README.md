# KIPL-ml

ML for WF.

### Setup:

Create a virtual env using [uv](https://docs.astral.sh/uv/):

```
uv init
source .venv/bin/activate

uv sync
cd rustbindings
maturin develop --release --uv
```

Maturin builds the rustbindings require running (only once when creating... these commands add necessary stuff to pyproject.toml..)

`uv init --lib PACKAGE`
`uv init --build-backend maturin example-ext`

### Data:

Download the "bigenough" dataset from [here](https://dart.cse.kau.se/maybenot/bigenough-95x10x20-standard-rngsubpages.tar.gz).

We rather have a single file for a trace to prepare for potential massive datasets.
In order to make the conversion to this standard use `kipl_ml/data/conversion.py --dataset DATASET`.
There are implementations for "bigenough" and "ts5" (deprecated) datasets.
In order to find these you need to set two paths in `.env` file.

```
.env

WF_DATA_DIR=/path/to/orig/data
MLFLOW_TRACKING_URI=http://127.0.0.1:8000
MACHINATION=/PATH/TO/MAYBENOT-GEN/target/release/machination
```

### MLFlow:

Run the mlflow server (note the env var from above):

`mlflow server --host 127.0.0.1 --port 8000 --backend-store-uri 'file:///abs/path/to/projectroot/.mlruns' --artifacts-destination '.mlartifacts'`

##### To train the laserbeak models you'll also need:

`git@github.com:notem/Laserbeak-WF-Classifier.git`

Unfortunately the above is hard to install so just add it into you pythonpath for now... e.g., modify the venv activate script by adding:

`export PYTHONPATH="$PYTHONPATH:/PATH/TO/Laserbeak-WF-Classifier"`

You also need:

`git@github.com:huggingface/pytorch-image-models.git`

however, thos come as depencies and do not require manual installation.


### PyDeps for dep tracking:

Run:

`pydeps path/to/module.py`

to generate a dependency graph of a module. Config file in `.pydeps`.

### Conventions:

From the `bigenough` we currently map the direction flags: "s"(end) and "r"(eceive) into 1 and -1 respectively. We consider dir > 0 as the "upload" direction (according to Laserbeak conventions). See `kipl_ml/data/conversion.py` for the mapping.

### TODO:

Bring attacks form [wf-lib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/).
