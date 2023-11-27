from pathlib import Path
import json
import time

import pandas as pd
import numpy as np

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
)
from pymoo.core.crossover import Crossover
from pymoo.core.callback import Callback
from pymoo.core.population import Population

from poli import objective_factory

from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem
from poli_baselines.core.utils.pymoo.wildtype_sampling import WildtypeMutationSampling
from poli_baselines.core.utils.pymoo.wildtype_mating import WildtypeMating
from poli_baselines.core.utils.pymoo.save_history import (
    _from_dict_to_list,
    _from_list_to_dict,
)


class SaveCallback(Callback):
    def __init__(self, saving_path: Path, save_every: int = None) -> None:
        self.saving_path = saving_path
        self.saving_path.mkdir(exist_ok=True)
        self.save_every = save_every
        super().__init__()

    def notify(self, algorithm: NSGA2):
        # We will save the population at each generation
        current_generation = {
            "x": [_from_dict_to_list(x) for x in algorithm.pop.get("X").tolist()],
            "y": [y for y in algorithm.pop.get("F").tolist()],
        }

        with open(self.saving_path / f"history_{algorithm.n_gen}.json", "w") as f:
            json.dump(current_generation, f)

        self.data[algorithm.n_gen] = current_generation

        return super().notify(algorithm)


class NoCrossover(Crossover):
    def __init__(self, n_parents=2, n_offsprings=2, prob=0.9, **kwargs):
        super().__init__(n_parents, n_offsprings, prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X


if __name__ == "__main__":
    # Loading up the initial Pareto front
    # to get the initial wildtypes
    THIS_DIR = Path(__file__).parent.resolve()
    original_pareto_front = pd.read_csv(
        THIS_DIR / "initial_pareto_front.csv",
        index_col=False,
    )

    START_FROM = None
    # START_FROM = THIS_DIR / "history" / "1694697851" / "history_00013.json"

    wildtype_pdb_paths = []
    for pdb_id in original_pareto_front["pdb_id"]:
        folder_to_pdb = list((THIS_DIR / "repaired_pdbs").glob(f"{pdb_id}_*"))
        if len(list(folder_to_pdb)) == 0:
            print(f"Could not find PDB for {pdb_id}")
            continue

        folder_to_pdb = list(folder_to_pdb)[0]
        pdb_id_and_chain = folder_to_pdb.name
        pdb_path = folder_to_pdb / f"{pdb_id_and_chain}_Repair.pdb"
        wildtype_pdb_paths.append(pdb_path)

    # Creating the objective function
    problem_info, f, x0, y0, _ = objective_factory.create(
        name="foldx_stability_and_sasa",
        wildtype_pdb_path=wildtype_pdb_paths,
        parallelize=True,
        num_workers=8,
        batch_size=8,
    )

    # Saving the wildtypes
    time_ = str(int(time.time()))
    history_dir = THIS_DIR / "history" / time_
    history_dir.mkdir(exist_ok=True, parents=True)
    with open(history_dir / "wildtype_scores.json", "w") as fp:
        json.dump({"x": x0.tolist(), "y": y0.tolist()}, fp)

    pymoo_problem = DiscretePymooProblem(
        black_box=-f,
        x0=x0,
        y0=y0,
    )

    # Now we can use PyMoo's NSGA-II to solve the problem.
    population_size = 8

    if START_FROM is not None:
        with open(START_FROM, "r") as fp:
            history = json.load(fp)

        X = np.array([_from_list_to_dict(x) for x in history["x"]])
        F = np.array(history["y"])
        sampling = Population.new("X", X, "F", F)
    else:
        sampling = WildtypeMutationSampling(
            x_0=x0, alphabet=problem_info.alphabet, num_mutations=5
        )

    method = NSGA2(
        pop_size=population_size,
        sampling=sampling,
        mating=WildtypeMating(num_mutations=4),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    # Now we can minimize the problem
    res = minimize(
        pymoo_problem,
        method,
        termination=("n_gen", 10),
        seed=1,
        save_history=True,
        verbose=True,
        callback=SaveCallback(history_dir),
    )