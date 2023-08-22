"""
In this example, we optimize the QED of molecules
as outputted by a VAE. The VAE is trained on a small
subset of PubChem, and the QED is calculated using poli.
"""
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn

from poli import objective_factory
from poli_baselines.solvers import LatentSpaceBayesianOptimization


THIS_DIR = Path(__file__).parent.resolve()


if __name__ == "__main__":
    # Loading the objective
    _, f_smb, _, _, _ = objective_factory.create(
        name="super_mario_bros",
    )
    x_0 = np.array([[1.0, 1.0]])
    y_0 = f_smb(x_0)

    # Defining the solver
    solver = LatentSpaceBayesianOptimization(
        black_box=f_smb,
        x0=x_0,
        y0=y_0,
        encoder=lambda x: x,
        decoder=lambda z: z,
    )

    solver.solve(max_iter=25, verbose=True)
    print(solver.history)
