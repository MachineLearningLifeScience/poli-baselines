import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from poli.objective_repository import RaspProblemFactory
from poli_baselines.solvers.bayesian_optimization.lambo2 import Lambo2

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from simple_observer import SimpleObserver, plot_best_y


def run_with_default_hyperparameters():
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

    lambo2 = Lambo2(
        black_box=black_box,
        x0=x0,
        y0=y0,
    )

    lambo2.solve(max_iter=10)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_best_y(observer, ax1)
    plot_best_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    plt.show()

    black_box.terminate()


def run_with_modified_hyperparameters():
    """
    TODO: write docs explaining the original config
    file from Lambo2.
    """
    # Some alternative hyperparameters which
    # will override what is specified in the Lambo2
    # configuration. You can find the original config
    # we use here:
    # TODO: add link.
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

    lambo2 = Lambo2(
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


if __name__ == "__main__":
    # run_with_default_hyperparameters()
    run_with_modified_hyperparameters()
