"""
In this script, we optimize the stability and SASA of a protein using pymoo's implementation of NSGA-II.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover

from poli import objective_factory

from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem
from poli_baselines.core.utils.pymoo.save_history import save_final_population

THIS_DIR = Path(__file__).parent.resolve()


class WildtypeMutationSampling(Sampling):
    def __init__(self, x_0: np.ndarray, num_mutations: int = 1) -> None:
        self.x_0 = x_0
        self.num_mutations = num_mutations
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        samples = np.repeat(self.x_0, n_samples, axis=0)
        minimum_ = problem.xl
        maximum_ = problem.xu

        # To perform different self.num_mutations, we first
        # select the indices to mutate.

        # Then we mutate the selected indices.
        for sample in samples:
            all_indices_at_random = np.random.permutation(samples.shape[1])
            indices_to_mutate = all_indices_at_random[: self.num_mutations]
            for idx in indices_to_mutate:
                sample[idx] = np.random.randint(minimum_[idx], maximum_[idx] + 1)

        return samples


class IntegerFlipMutation(Mutation):
    def __init__(self, num_mutations=1, prob=1, prob_var=None, **kwargs) -> None:
        self.num_mutations = num_mutations
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        minimum_ = problem.xl
        maximum_ = problem.xu
        Xp = np.copy(X)

        # To perform different self.num_mutations, we first
        # select the indices to mutate.
        all_indices_at_random = np.random.permutation(X.shape[1])
        indices_to_mutate = all_indices_at_random[: self.num_mutations]

        # Then we mutate the selected indices.
        for xp in Xp:
            for idx in indices_to_mutate:
                xp[idx] = np.random.randint(minimum_[idx], maximum_[idx] + 1)

        return Xp


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
        black_box=-f_stability_and_sasa,
        alphabet=problem_info.alphabet,
        x0=x_0,
        y0=y_0,
    )

    # Now we can use PyMoo's NSGA-II to solve the problem.
    # The cross-over and mutation are defined in such a way
    # that they manipulate integers instead of floats.
    # See e.g. https://pymoo.org/customization/discrete.html
    method = NSGA2(
        pop_size=10,
        sampling=WildtypeMutationSampling(x_0=x_0, num_mutations=1),
        crossover=UniformCrossover(),
        mutation=IntegerFlipMutation(num_mutations=1),
        eliminate_duplicates=True,
    )

    # Now we can minimize the problem
    res = minimize(problem, method, termination=("n_gen", 2), seed=1, save_history=True)

    # And print the results
    if len(res.X.shape) == 1:
        best_solution = res.X
    else:
        best_solution = res.X[0]
    inverse_alphabet = {v: k for k, v in problem_info.alphabet.items()}
    best_solution_as_string = "".join([inverse_alphabet[i] for i in best_solution])
    print(f"Nr. good solutions found: {len(res.X)}")
    print(f"Best solution found (index 0): {res.X[0]} ({best_solution_as_string})")
    print(f"Function value: {-res.F}")
    print(f"Starting value: {y_0}")

    save_final_population(
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
    sns.scatterplot(x=y_0[:, 0], y=y_0[:, 1], ax=ax, label="Wildtype", c="red", marker="x")
    ax.set_xlabel("Stability")
    ax.set_ylabel("SASA")

    plt.show()
