import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> os.PathLike:
    kipl_ml =  os.path.dirname(os.path.abspath(__file__))
    if not kipl_ml.endswith("kipl_ml"):
        raise ValueError("This file should be in the kipl_ml package")
    return os.path.dirname(kipl_ml)

ROOT_DIR = get_project_root()


try:
    plt.style.use(os.path.join(ROOT_DIR, ".config", "pltstyle.mplstyle"))
except FileNotFoundError:
    plt.style.use("seaborn-darkgrid")
