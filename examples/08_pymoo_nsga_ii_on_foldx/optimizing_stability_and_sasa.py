"""
In this script, we optimize the stability and SASA of a protein using pymoo's implementation of NSGA-II.
"""
from typing import List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)

from poli import objective_factory

from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem
from poli_baselines.core.utils.pymoo.save_history import save_all_populations

THIS_DIR = Path(__file__).parent.resolve()


class WildtypeMutationSampling(Sampling):
    """
    TODO: document
    """

    def __init__(
        self, x_0: np.ndarray, alphabet: List[str], num_mutations: int = 1
    ) -> None:
        """
        TODO: document
        """
        # We assume a shape [b, L], where some of the rows
        # have been padded with empty strings.
        self.x_0 = x_0
        self.num_mutations = num_mutations
        self.alphabet = alphabet
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        """
        This mutation takes a random wildtype (i.e. a row
        of self.x_0), and mutates it at a random place
        self.num_mutations times.

        To comply with what pymoo expects for discrete
        (choice) problems, This needs to output a list
        of dictionaries, where each dictionary has keys
        "x_0", "x_1", etc. and values that are elements
        of the alphabet.
        """
        # Selecting a row from self.x_0 at random
        random_wildtype = np.random.choice(self.x_0, size=1)

        # Cleaning up the padding
        random_wildtype = random_wildtype[random_wildtype != ""]
        sequence_length = random_wildtype.shape[1]

        # Defining the mutations
        mutations = []
        for _ in range(n_samples):
            # The original dict
            mutation = {f"x_{i}": random_wildtype[0, i] for i in range(sequence_length)}
            all_indices_at_random = np.random.permutation(sequence_length)
            indices_to_mutate = all_indices_at_random[: self.num_mutations]

            # Then we mutate the selected indices.
            for idx in indices_to_mutate:
                mutation[f"x_{idx}"] = np.random.choice(self.alphabet)

            # And we add it to the list of mutations
            mutations.append(mutation)

        return mutations


class NoMutation(Mutation):
    def __init__(self, prob=1, prob_var=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X


if __name__ == "__main__":
    # Creating e.g. the aloha problem. We require that the black
    # box objective function takes integers as inputs.
    path_to_wildtype = THIS_DIR / "3ned_Repair.pdb"

    # Creating a problem for computing both stability and SASA at the
    # same time.
    problem_info, f_stability_and_sasa, x_0, y_0, _ = objective_factory.create(
        name="foldx_stability_and_sasa", wildtype_pdb_path=path_to_wildtype
    )

    # Since PyMoo is used to minimizing instead of maximizing (our convention),
    # we pass -f instead of f.
    problem = DiscretePymooProblem(
        black_box=f_stability_and_sasa,
        x0=x_0,
        y0=y_0,
    )

    # Now we can use PyMoo's NSGA-II to solve the problem.
    # The cross-over and mutation are defined in such a way
    # that they manipulate integers instead of floats.
    # See e.g. https://pymoo.org/customization/discrete.html
    method = NSGA2(
        pop_size=10,
        sampling=WildtypeMutationSampling(
            x_0=x_0, alphabet=problem_info.alphabet, num_mutations=1
        ),
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            mutation={Choice: NoMutation()},
        ),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    # Now we can minimize the problem
    res = minimize(problem, method, termination=("n_gen", 3), seed=1, save_history=True)

    save_all_populations(
        result=res,
        alphabet=problem_info.alphabet,
        path=THIS_DIR / "history.json",
    )

    # Let's plot all the different populations:
    all_F = -np.concatenate(
        [history_i.pop.get("F") for history_i in res.history], axis=0
    )
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        x=all_F[:, 0],
        y=all_F[:, 1],
        ax=ax,
        label="All populations",
    )
    sns.scatterplot(
        x=y_0[:, 0], y=y_0[:, 1], ax=ax, label="Wildtype", c="red", marker="x"
    )
    ax.set_xlabel("Stability")
    ax.set_ylabel("SASA")

    plt.show()
