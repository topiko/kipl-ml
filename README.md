# KIPL-ml

ML for WF.

### Setup:

Create a virtual env:

`virtualenv -p /your/python3.10 venv-kipl-ml`

Then activate, install, and get the requirements from `requirements.txt`.

```
pip install -e .
pip install -r requirements.txt
```

### Data:

We rather have a single file for a trace to prepare for potential massive datasets.
In order to make the conversion to this standard use `kipl_ml/data/conversion.py --dataset DATASET`.
There are implementations for "bigenough" and "ts5" datasets.
In order to find these you need to set two paths in `.env` file.

```
.env

WF_DATA_DIR=/path/to/orig/data
STD_FLOWS_DATA_DIR=/where/you/want/to/save
MLFLOW_TRACKING_URI=http://127.0.0.1:8000
```

### MLFlow:

Run the mlflow server (note the env var from above):

`mlflow server --host 127.0.0.1 --port 8000 --backend-store-uri 'file:///abs/path/to/projectroot/.mlruns'`

##### To train the laserbeak models you'll also need:

`git@github.com:notem/Laserbeak-WF-Classifier.git`

Modify the venv activate script by adding:

`export PYTHONPATH="$PYTHONPATH:/PATH/TO/Laserbeak-WF-Classifier"`

You also need to clone and install (for laserbeak to work):

`git@github.com:huggingface/pytorch-image-models.git`


### PyDeps for dep tracking:

Run:

`pydeps path/to/module.py`

to generate a dependency graph of a module. Config file in `.pydeps`.
