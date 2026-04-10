#!/bin/bash

uv run python -m unittest kipl_ml.metrics.tests kipl_ml.rl.tests kipl_ml.trace.tests kipl_ml.models.tests -v
