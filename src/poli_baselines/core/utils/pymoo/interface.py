"""
This module contains ways of transforming poli's objective
functions to pymoo problems.
"""
from typing import Dict

import numpy as np

from pymoo.core.problem import (
    ElementwiseEvaluationFunction,
    LoopedElementwiseEvaluation,
    Problem,
)
from pymoo.core.variable import Choice

from poli.core.multi_objective_black_box import MultiObjectiveBlackBox


class DiscretePymooProblem(Problem):
    def __init__(
        self,
        black_box: MultiObjectiveBlackBox,
        alphabet: Dict[str, int],
        x0: np.ndarray,
        y0: np.ndarray,
        **kwargs,
    ):
        self.x0 = x0
        self.y0 = y0
        self.alphabet = alphabet

        super().__init__(
            n_var=x0.shape[1],
            n_obj=y0.shape[1],
            xl=np.zeros(x0.shape[1]),
            xu=np.ones(x0.shape[1]) * (len(alphabet) - 1),
            vtype=int,
            **kwargs,
        )
        self.black_box = black_box

    def _evaluate(self, x, out, *args, **kwargs):
        # an [b, n] array, where n is the number of objectives
        f = self.black_box(x, context=kwargs.get("context", None))
        out["F"] = f


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

    # Creating e.g. the aloha problem. We require that the black
    # box objective function takes integers as inputs.
    problem_info, f_aloha, x_0_aloha, _, _ = objective_factory.create(name="aloha")

    # Creating a multi-objective black box using two copies
    # of aloha
    f = MultiObjectiveBlackBox(L=5, objective_functions=[f_aloha, f_aloha])
    y_0 = f(x_0_aloha)

    # Since PyMoo is used to minimizing instead of maximizing (our convention),
    # we pass -f instead of f.
    problem = DiscretePymooProblem(
        black_box=-f,
        alphabet=problem_info.alphabet,
        x0=x_0_aloha,
        y0=y_0,
    )

    # Now we can use PyMoo's NSGA-II to solve the problem.
    # The cross-over and mutation are defined in such a way
    # that they manipulate integers instead of floats.
    # See e.g. https://pymoo.org/customization/discrete.html
    method = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    # Now we can minimize the problem
    res = minimize(
        problem, method, termination=("n_gen", 50), seed=1, save_history=True
    )

    # And print the results
    best_solution = res.X[0]
    inverse_alphabet = {v: k for k, v in problem_info.alphabet.items()}
    best_solution_as_string = [inverse_alphabet[i] for i in best_solution]
    print(f"Best solution found: {res.X} ({best_solution_as_string})")
    print(f"Function value: {res.F}")
