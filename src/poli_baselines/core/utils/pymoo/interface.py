"""
This module contains ways of transforming poli's objective
functions to pymoo problems.
"""
from typing import Dict, List, Tuple
from pathlib import Path
import pickle

import numpy as np

from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.variable import Choice
from pymoo.core.population import Population

from poli.core.multi_objective_black_box import MultiObjectiveBlackBox


class DiscretePymooProblem(Problem):
    def __init__(
        self,
        black_box: MultiObjectiveBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        checkpoint_path: Path = None,
        **kwargs,
    ):
        """
        TODO: Document
        """
        self.x0 = x0
        self.y0 = y0

        # We define sequence_length discrete choice variables,
        # selecting from the alphabet, which we assure is List[str]
        # (by checking the first key, if it's dict).
        alphabet = black_box.info.alphabet
        if isinstance(alphabet, Dict):
            alphabet = list(alphabet.keys())
            assert isinstance(alphabet[0], str)

        sequence_length = x0.shape[1]
        variables = {f"x_{i}": Choice(options=alphabet) for i in range(sequence_length)}

        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            X, F = self._load_checkpoint()
            pop = Population.new("X", X)
            pop.set("F", F)
        else:
            pop = None

        super().__init__(
            vars=variables,
            n_obj=y0.shape[1],
            sampling=pop,
            **kwargs,
        )
        self.black_box = black_box

    def _from_dict_to_array(self, x: List[dict]) -> np.ndarray:
        """
        Transforms a list of dicts {x_i: value}
        to an array of shape [n, sequence_length]
        """
        if isinstance(x, dict):
            new_x = np.array([x[f"x_{i}"] for i in range(len(x))])
        else:
            new_x = np.array([[x_[f"x_{i}"] for i in range(len(x_))] for x_ in x])

        return new_x

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the black box function by transforming
        the discrete choices to a vector, and then evaluating.
        """
        # At this point, x is a dictionary with keys "x_0", "x_1", etc.
        # and we assume that were evaluating a single x at a time.
        # TODO: is there a way to parallelize? To implement this,
        # using Problem instead of ElementwiseProblem might be necessary.
        x = self._from_dict_to_array(x)
        # x = np.array([x[f"x_{i}"] for i in range(len(x))]).reshape(1, -1)

        # The output is a [1, n] array, where n is the number of objectives
        f = self.black_box(x, context=kwargs.get("context", None))
        out["F"] = f

    def save_checkpoint(self, saving_path: Path):
        X = self.pop.get("X")
        F = self.pop.get("F")

        with open(saving_path, "wb") as fp:
            pickle.dump((X, F), fp)

    def _load_checkpoint(self) -> Tuple[dict, list]:
        with open(self.checkpoint_path, "rb") as fp:
            X, F = pickle.load(fp)

        return X, F


if __name__ == "__main__":
    """
    The following is an example of how to register and use
    this DiscretePymooProblem using poli's objective functions.
    """
    from poli import objective_factory
    from poli.core.multi_objective_black_box import MultiObjectiveBlackBox

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    from pymoo.core.mixed import (
        MixedVariableMating,
        MixedVariableSampling,
        MixedVariableDuplicateElimination,
    )

    # Creating e.g. the aloha problem. We require that the black
    # box objective function takes integers as inputs.
    problem_info, f_aloha, x_0_aloha, _, _ = objective_factory.create(name="aloha")

    # Creating a multi-objective black box using two copies
    # of aloha
    f = MultiObjectiveBlackBox(
        info=problem_info, objective_functions=[f_aloha, f_aloha]
    )
    y_0 = f(x_0_aloha)

    # Since PyMoo is used to minimizing instead of maximizing (our convention),
    # we pass -f instead of f.
    problem = DiscretePymooProblem(
        black_box=-f,
        x0=x_0_aloha,
        y0=y_0,
    )

    # Now we can use PyMoo's NSGA-II to solve the problem.
    # The cross-over and mutation are defined in such a way
    # that they manipulate integers instead of floats.
    # See e.g. https://pymoo.org/customization/discrete.html
    method = NSGA2(
        pop_size=100,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination()
        ),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    # Now we can minimize the problem
    res = minimize(
        problem,
        method,
        termination=("n_gen", 50),
        seed=1,
        save_history=True,
        verbose=True,
    )

    # And print the results
    best_solution = res.X[0]
    best_solution_as_string = "".join(
        [best_solution[f"x_{i}"] for i in range(len(best_solution))]
    )
    print(f"Best solution found: {res.X} ({best_solution_as_string})")
    print(f"Function value: {res.F}")
