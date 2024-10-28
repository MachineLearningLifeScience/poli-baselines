import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from poli.objective_repository import RaspProblemFactory

from poli_baselines.solvers.simple.random_mutation import RandomMutation

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from simple_observer import SimpleObserver, plot_best_y  # noqa: E402


def run_with_default_hyperparameters():
    from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

    RFP_PDBS_DIR = THIS_DIR / "rfp_pdbs"
    ALL_PDBS = list(RFP_PDBS_DIR.rglob("**/*.pdb"))
    problem = RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=[p.parent.name.split("_")[1] for p in ALL_PDBS],
    )
    black_box = problem.black_box

    observer = SimpleObserver()
    black_box.set_observer(observer)

    arr = np.load(THIS_DIR / "rasp_seed_data.npz")
    x0 = arr["x0"]
    y0 = arr["y0"]

    observer.x_s.append(x0.reshape(-1, 1))
    observer.y_s.append(y0)

    torch.set_float32_matmul_precision("medium")
    lambo2 = LaMBO2(
        black_box=black_box,
        x0=x0,
        y0=y0,
        overrides=["max_epochs=2"],
        max_epochs_for_retraining=8,
    )
    lambo2.solve(max_iter=32)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_best_y(observer, ax1)
    plot_best_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    plt.show()

    print("Best starting obj value: ", np.max(y0))
    print("Best final obj value: ", np.max(lambo2.history_for_training["y"]))

    black_box.terminate()


def run_with_modified_hyperparameters():
    """
    In this example we modify the configuration
    of the optimizer, which uses hydra underneath.

    In the overrides kwarg of LaMBO2, we can specify
    the hyperparameters we want to change. For example,
    we can change the population size, the number of
    epochs for pretraining, etc.

    You can find the original configuration we use here:
    src/poli_baselines/solvers/bayesian_optimization/lambo2/hydra_configs
    """
    from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

    POPULATION_SIZE = 96
    MAX_EPOCHS_FOR_PRETRAINING = 4

    RFP_PDBS_DIR = THIS_DIR / "rfp_pdbs"
    ALL_PDBS = list(RFP_PDBS_DIR.rglob("**/*.pdb"))
    problem = RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=[p.parent.name.split("_")[1] for p in ALL_PDBS],
    )
    black_box = problem.black_box

    observer = SimpleObserver()
    black_box.set_observer(observer)

    arr = np.load(THIS_DIR / "rasp_seed_data.npz")
    x0 = arr["x0"]
    y0 = arr["y0"]

    observer.x_s.append(x0.reshape(-1, 1))
    observer.y_s.append(y0)

    lambo2 = LaMBO2(
        black_box=black_box,
        x0=x0,
        y0=y0,
        overrides=[
            f"num_samples={POPULATION_SIZE}",
            f"max_epochs={MAX_EPOCHS_FOR_PRETRAINING}",
        ],
    )

    lambo2.solve(max_iter=4)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_best_y(observer, ax1)
    plot_best_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    plt.show()

    black_box.terminate()


def comparing_against_directed_evolution():
    arr = np.load(THIS_DIR / "rasp_seed_data.npz")
    x0 = arr["x0"]
    y0 = arr["y0"]
    batch_size = 128
    n_iterations = 32

    x0_for_solver_no_padding = x0[np.argsort(y0.flatten())[::-1]][:batch_size]

    # Adding padding
    max_length = max(map(len, x0_for_solver_no_padding))
    x0_for_solver_ = [[char for char in x_i] for x_i in x0_for_solver_no_padding]
    x0_for_solver = np.array(
        [list_ + ([""] * (max_length - len(list_))) for list_ in x0_for_solver_]
    )

    y0_for_solver = y0[np.argsort(y0.flatten())[::-1]][:batch_size]

    RFP_PDBS_DIR = THIS_DIR / "rfp_pdbs"
    ALL_PDBS = list(RFP_PDBS_DIR.rglob("**/*.pdb"))
    problem = RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=[p.parent.name.split("_")[1] for p in ALL_PDBS],
    )
    black_box = problem.black_box

    observer = SimpleObserver()
    black_box.set_observer(observer)

    observer.x_s.append(x0.reshape(-1, 1))
    observer.y_s.append(y0)

    directed_evolution = RandomMutation(
        black_box=black_box,
        x0=x0_for_solver,
        y0=y0_for_solver,
        batch_size=batch_size,
    )
    max_eval = n_iterations * batch_size
    directed_evolution.solve(max_iter=max_eval // batch_size, verbose=True)
    observer.save_history(
        THIS_DIR / f"directed_evolution_rasp_trace_b_{batch_size}_i_{n_iterations}.npz"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_best_y(observer, ax1)
    plot_best_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    plt.show()

    black_box.terminate()


if __name__ == "__main__":
    # run_with_default_hyperparameters()
    # run_with_modified_hyperparameters()
    comparing_against_directed_evolution()
