"""
Implements a genetic algorithm solver using pymoo as a backend.
"""

from typing import Callable, Iterable
from typing_extensions import Self
from collections import defaultdict
import logging

import numpy as np
from numpy import ndarray
from pymoo.core.mixed import (
    MixedVariableGA,
    MixedVariableSampling,
    MixedVariableMating,
    MixedVariableDuplicateElimination,
)
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.variable import Choice
from pymoo.core.individual import Individual
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.core.selection import Selection

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.utils.pymoo.interface import (
    DiscretePymooProblem,
    _from_dict_to_array,
    _from_array_to_dict,
)
from poli_baselines.core.utils.mutations import add_random_mutations_to_reach_pop_size

from poli_baselines.core.abstract_solver import AbstractSolver


class SaveHistoryCallback(Callback):
    def __init__(self, solver: AbstractSolver):
        super().__init__()
        self.solver = solver

    def notify(self, algorithm):
        # Since we're dealing with MixedVariables, we need to convert the
        # population of dicts {"x_i": value} to an array of shape [n, sequence_length].
        x_as_array = np.vstack(
            [_from_dict_to_array(x_i) for x_i in algorithm.pop.get("X")]
        )

        self.solver.update(x_as_array, -algorithm.pop.get("F"))


class RandomSelectionOfSameLength(Selection):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, _, pop, n_select, n_parents, **kwargs):
        # TODO: implement this by sub-selecting the parents that have a certain length, and forming pairs thereof.

        # This dict will be of the form {length: [index_of_pop_element]},
        # separating the elements of the population according
        # to their length
        subpopulations_of_same_length = defaultdict(list)
        for i, individual in enumerate(pop):
            individuals_length = len([v for v in individual.x.values() if v != ""])
            subpopulations_of_same_length[individuals_length].append(i)

        # For each n_select, we randomly choose a length and select two
        # random parents from said subset.
        # TODO: What should we do if there's only one element of a certain length?
        parents_ = []
        for _ in range(n_select):
            random_length = np.random.choice(list(subpopulations_of_same_length.keys()))
            index_1, index_2 = np.random.choice(
                subpopulations_of_same_length[random_length],
                size=2,
            )
            parents_.append((index_1, index_2))

        return np.array(parents_, dtype=int)


class FixedLengthGeneticAlgorithm(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: ndarray,
        y0: ndarray,
        population_size: int = 100,
        initialize_with_x0: bool = True,
        prob_of_mutation: float = 0.5,
    ):
        super().__init__(black_box, x0, y0)
        self.population_size = population_size
        # TODO: make sure that the padding is not in the alphabet.

        self.pymoo_problem = DiscretePymooProblem(
            black_box=self.black_box,
            x0=self.x0,
            y0=self.y0,
            initialize_with_x0=initialize_with_x0,
        )

        if initialize_with_x0:
            initial_population = self._compute_initial_population_from_x0()
            sampling = initial_population
        else:
            sampling = MixedVariableSampling()

        self.optimizer = MixedVariableGA(
            pop_size=population_size,
            sampling=sampling,
            mating=MixedVariableMating(
                mutation={
                    Choice: ChoiceRandomMutation(prob_var=prob_of_mutation),
                },
                selection=RandomSelectionOfSameLength(),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
            ),
        )

    def next_candidate(self) -> ndarray:
        raise NotImplementedError

    def solve(
        self,
        max_iter: int = 100,
        break_at_performance: float = None,
        verbose: bool = False,
        pre_step_callbacks: Iterable[Callable[[Self], None]] = None,
        post_step_callbacks: Iterable[Callable[[Self], None]] = None,
    ) -> ndarray:
        res = minimize(
            self.pymoo_problem,
            self.optimizer,
            ("n_gen", max_iter),
            verbose=verbose,
            callback=SaveHistoryAndCallOtherCallbacks(self, post_step_callbacks),
        )

        return _from_dict_to_array([res.X]), -res.F

    def _compute_initial_population_from_x0(self) -> Population:
        # Pad x_0/subsample x_0 depending on pop_size
        x0 = self.x0
        y0 = self.y0
        pop_size = self.population_size

        x0_for_initialization = None
        if x0.shape[0] > pop_size:
            # Subsample.
            logging.warning(
                "Warning: initializing with an x0 that is larger than the population size. We will subsample the best performing starting points"
            )

            # TODO: this assumes a single objective.
            best_performing_indices = np.argsort(y0.flatten())[::-1]
            x0_for_initialization = x0[best_performing_indices[:pop_size]]
            y0_for_initialization = y0[best_performing_indices[:pop_size]]
        elif x0.shape[0] < pop_size:
            # Pad using random mutations (?).
            x0_for_initialization = add_random_mutations_to_reach_pop_size(
                x0, self.black_box.info.alphabet, pop_size
            )
            missing_evaluations = self.black_box(x0_for_initialization[x0.shape[0] :])
            y0_for_initialization = np.vstack([y0, missing_evaluations])
        else:
            # We're golden.
            x0_for_initialization = x0
            y0_for_initialization = y0

        # TODO: Will this be enough?
        x0_for_initialization_as_dicts = _from_array_to_dict(x0_for_initialization)
        initial_individuals = [
            Individual(X=x_i, F=y_i)
            for (x_i, y_i) in zip(
                x0_for_initialization_as_dicts, y0_for_initialization.flatten()
            )
        ]

        return Population(individuals=initial_individuals)


class SaveHistoryAndCallOtherCallbacks(Callback):
    def __init__(
        self, solver: FixedLengthGeneticAlgorithm, callbacks: Iterable[Callback]
    ):
        super().__init__()
        self.solver = solver
        self.callbacks = callbacks

    def notify(self, algorithm):
        x_as_array = np.vstack(
            [_from_dict_to_array(x_i) for x_i in algorithm.pop.get("X")]
        )

        self.solver.update(x_as_array, -algorithm.pop.get("F"))

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self.solver)


if __name__ == "__main__":
    from poli.core.util.seeding import seed_python_numpy_and_torch
    from poli.objective_repository import AlohaProblemFactory

    seed_python_numpy_and_torch(13)

    f, x0, y0 = AlohaProblemFactory().create()

    solver = FixedLengthGeneticAlgorithm(
        black_box=f, x0=x0, y0=y0, population_size=10, initialize_with_x0=True
    )

    x = solver.solve(max_iter=100, verbose=False)
    print(x)
    print(solver.get_history_as_arrays()[0])
