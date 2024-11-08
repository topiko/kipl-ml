import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.getenv("KIPL_ML_ROOT")


try:
    plt.style.use(os.path.join(ROOT_DIR, ".config", "pltstyle.mplstyle"))
except FileNotFoundError:
    plt.style.use("seaborn-darkgrid")
