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

We store metadata and split info in a standardized `metadf.h5` per dataset.

You **must run conversion before training/evaluation scripts** (and re-run it when conversion schema changes).

Use:

```bash
python kipl_ml/data/conversion.py --dataset bigenough
python kipl_ml/data/conversion.py --dataset gong-surakav
```

This command:

- parses raw `.log` traces under `WF_DATA_DIR/<dataset>/...`
- writes standardized metadata to `.data/<dataset>/metadf.h5`
- regenerates XV split columns in that metadata file

There are implementations for `bigenough` and `gong-surakav`.

Important: if you pull new changes to `kipl_ml/data/conversion.py` (for example new metadata columns like `time_to_<X>_packets`), re-run conversion so your local `metadf.h5` is up to date.

This repo uses a repo-root `.env` file for paths and MLflow settings.

### Environment variables (.env)

Most entrypoints call `dotenv.load_dotenv()` and expect a `.env` file in the repo root.

Required (depending on what you run):

- `WF_DATA_DIR`: path to the directory containing the original dataset folders (used by `kipl_ml/data/conversion.py`).
- `MLFLOW_TRACKING_URI`: required by most experiment scripts (e.g. `experiment/ephemeral_defences/main.py` asserts it is set).
- `MAYBENOT`: path to the `maybenot` binary (required for fixed-machine defences: Breakpad/FRONT/Interspace/Regulator/Tamaraw).
- `MACHINATION`: **deprecated**, use `MAYBENOT` instead.

Optional:

- `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`: only if your MLflow server requires authentication.

```
.env

WF_DATA_DIR=/path/to/orig/data
MAYBENOT=/PATH/TO/MAYBENOT/target/release/maybenot

MLFLOW_TRACKING_URI=http://127.0.0.1:8000
MLFLOW_TRACKING_USERNAME=***
MLFLOW_TRACKING_PASSWORD=***
```

### MLFlow:

Run the mlflow server (note the env var from above) - ONLY if you don't have access to external server:

`mlflow server --host 127.0.0.1 --port 8000 --backend-store-uri 'file:///abs/path/to/projectroot/.mlruns' --artifacts-destination '.mlartifacts'`

##### To train the laserbeak models you'll also need(?):

`git@github.com:topiko/laserbeak.git`
but it is listed as an dep. and should work out of the box.

### PyDeps for dep tracking:

Run:

`pydeps path/to/module.py`

to generate a dependency graph of a module. Config file in `.pydeps`.

### Conventions:

From the `bigenough` we currently map the direction flags: "s"(end) and "r"(eceive) into 1 and -1 respectively. We consider dir > 0 as the "upload" direction (according to Laserbeak conventions). See `kipl_ml/data/conversion.py` for the mapping.

### For deterministic behavior:

`export CUBLAS_WORKSPACE_CONFIG=:4096:8`
