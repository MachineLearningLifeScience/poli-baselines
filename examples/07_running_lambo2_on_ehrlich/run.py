import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from poli.objective_repository import EhrlichProblemFactory
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from poli_baselines.solvers.simple.genetic_algorithm import FixedLengthGeneticAlgorithm

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from simple_observer import SimpleObserver, plot_best_y


def run_with_default_hyperparameters():
    problem = EhrlichProblemFactory().create(
        sequence_length=32,
        motif_length=4,
        n_motifs=2,
        quantization=4,
        return_value_on_unfeasible=-1.0
    )
    black_box = problem.black_box
    x0 = problem.x0
    random_seqs = np.array([list(black_box._sample_random_sequence()) for _ in range(127)])
    x0 = np.concatenate([problem.x0, random_seqs], axis=0)
    y0 = black_box(x0)

    observer = SimpleObserver()
    black_box.set_observer(observer)

    # arr = np.load(THIS_DIR / "rasp_seed_data.npz")
    # x0 = arr["x0"]
    # y0 = arr["y0"]

    observer.x_s.append(x0.reshape(-1, 1))
    observer.y_s.append(y0)

    presolver = FixedLengthGeneticAlgorithm(
        black_box=black_box,
        x0=x0,
        y0=y0,
        population_size=128,
        prob_of_mutation=0.4
    )
    presolver.solve(max_iter=1)
    presolver_x = np.array(presolver.history["x"])
    presolver_x = presolver_x.reshape(presolver_x.shape[0], -1)

    # import pdb; pdb.set_trace()
    torch.set_float32_matmul_precision('medium')
    lambo2 = LaMBO2(
        black_box=black_box,
        x0=presolver_x,  # inconsistent API
        overrides=["max_epochs=2"],
        max_epochs_for_retraining=8,
    )

    lambo2.solve(max_iter=32)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_best_y(observer, ax1)
    plot_best_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    plt.show()

    black_box.terminate()


if __name__ == "__main__":
    run_with_default_hyperparameters()
