"""Implements NSGA-II as a poli-baselines solver.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective
genetic algorithm that uses a non-dominated sort to rank individuals and
a crowding distance to maintain diversity (TODO: fact-check and cite).

We leverage the implementation inside pymoo, and our wrapper for discrete
problems s.t. they can be used as pymoo Problems.
"""
from typing import Tuple, Type

import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.mating import InfillCriterion
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.termination import NoTermination
from pymoo.core.population import Population
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.variable import Choice

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem
from poli_baselines.core.utils.pymoo import (
    DiscreteSequenceSampling,
    DiscreteSequenceMating,
    DiscreteSequenceMutation,
    NoMutation,
    _from_dict_to_array,
    _from_array_to_dict,
)


class Discrete_NSGA_II(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        population_size: int = 10,
        initialize_with_x0: bool = True,
        mutation: Mutation = None,
        sampling: Sampling = None,
        mating: InfillCriterion = None,
    ):
        super().__init__(black_box, x0, y0)

        assert x0.shape[0] == population_size, (
            f"Expected x0 to have shape[0]=={population_size}, "
            f"but got {x0.shape[0]}."
        )

        self.pymoo_problem = DiscretePymooProblem(
            black_box=self.black_box,
            x0=self.x0,
            y0=self.y0,
            initialize_with_x0=initialize_with_x0,
        )

        # if mutation is None:
        #     mutation = MixedVariableMutation()

        if sampling is None:
            sampling = MixedVariableSampling()

        if mating is None:
            mating = MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            )

        termination = NoTermination()

        if initialize_with_x0:
            x0_as_mixed_variable = _from_array_to_dict(self.x0)
            sampling = Population.new("X", x0_as_mixed_variable)

        self.algorithm = NSGA2(
            pop_size=population_size,
            sampling=sampling,
            mating=mating,
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        self.algorithm.setup(
            problem=self.pymoo_problem,
            termination=termination,
            verbose=False,
        )

    def next_candidate(self) -> np.ndarray:
        # At this point, the pymoo problem has been initialized with a certain
        # population.
        x = self.algorithm.pop.get("X")
        return x

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        https://pymoo.org/algorithms/usage.html#Problem-Depdendent
        """
        pop = self.algorithm.ask()

        # Following the documentation of pymoo, we evaluate the
        # population on the problem
        self.algorithm.evaluator.eval(self.pymoo_problem, pop)

        # We tell the algorithm what has changed in the population.
        self.algorithm.tell(infills=pop)

        # We get the current population and objective values
        x = self.algorithm.pop.get("X")
        y = -self.algorithm.pop.get("F")  # minus, since we maximize

        # We update the history
        self.update(x, y)
        self.iteration += 1

        return x, y
