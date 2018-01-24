import numpy as np
from matplotlib import pyplot as plt
from itertools import product


n_iter = np.arange(10, 100, 10)
hyper_params = {"n_iter": n_iter,
                "tol": [1e-2, 1e-3, 1e-4],
                "n_components": [4,5,6]}
for i in hyper_params["n_components"]:
    print(i)