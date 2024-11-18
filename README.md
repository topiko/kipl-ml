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
