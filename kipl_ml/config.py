import os
from pathlib import Path

import kipl_ml

PROJECT_ROOT = Path(
    os.getenv("KIPL_ML_PROJECT_ROOT", Path(kipl_ml.__file__).parent.parent)
).resolve()
