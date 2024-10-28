import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from poli.objective_repository import FoldXStabilityProblemFactory

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from simple_observer import SimpleObserverForMads, plot_best_y  # noqa: E402


@click.command()
@click.option(
    "--num-workers", default=8, help="Number of workers to use for parallelization."
)
def run_with_modified_hyperparameters(num_workers: int):
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

    POPULATION_SIZE = 4
    MAX_EPOCHS_FOR_PRETRAINING = 1
    NUM_BATCHES = 2
    NUM_MUTATIONS_PER_STEP = 5

    arr = np.load(THIS_DIR / "dnja_foldx_initsamples.npz", allow_pickle=True)
    x0 = arr["x0"]
    y0 = arr["y0"]

    PDB_FILE = THIS_DIR / "repaired_pdbs" / "DNJA1_HUMAN_Repair.pdb"

    problem = FoldXStabilityProblemFactory().create(
        wildtype_pdb_path=PDB_FILE,
        parallelize=True,
        tmp_folder=THIS_DIR / "tmp",
        batch_size=POPULATION_SIZE,
        num_workers=num_workers,
    )
    black_box = problem.black_box

    observer = SimpleObserverForMads()
    black_box.set_observer(observer)

    observer.x_s.append(x0.reshape(-1, 1))
    observer.y_s.append(y0)

    first_candidate_points = x0[np.argsort(y0.flatten())[::-1]][:POPULATION_SIZE]

    lambo2 = LaMBO2(
        black_box=black_box,
        x0=x0,
        y0=y0,
        overrides=[
            f"num_samples={POPULATION_SIZE}",
            f"max_epochs={MAX_EPOCHS_FOR_PRETRAINING}",
            f"num_mutations_per_step={NUM_MUTATIONS_PER_STEP}",
            "random_seed=null",
        ],
        restrict_candidate_points_to=first_candidate_points,
    )

    lambo2.solve(max_iter=NUM_BATCHES)

    observer.save_history(THIS_DIR / "lambo2_trace.npz")
    observer.save_df_for_mads(THIS_DIR / "lambo2_trace.csv", batch_size=POPULATION_SIZE)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_best_y(observer, ax1)
    plot_best_y(observer, ax2, start_from=x0.shape[0])
    ax1.axvline(x0.shape[0], color="red")
    plt.show()

    black_box.terminate()


if __name__ == "__main__":
    run_with_modified_hyperparameters()
