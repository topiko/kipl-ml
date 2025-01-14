import os

import matplotlib.pyplot as plt

dir_ = os.path.dirname(os.path.abspath(__file__))

try:
    plt.style.use(os.path.join(dir_, "pltstyle.mplstyle"))
except FileNotFoundError:
    plt.style.use("seaborn-darkgrid")
