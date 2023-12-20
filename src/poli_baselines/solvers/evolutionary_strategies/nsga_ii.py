"""Implements NSGA-II as a poli-baselines solver.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective
genetic algorithm that uses a non-dominated sort to rank individuals and
a crowding distance to maintain diversity [1].

We leverage the implementation inside pymoo, and our wrapper for discrete
problems s.t. they can be used as pymoo Problems.

References:
-----------
[1] Deb, K., A. Pratap, S. Agarwal, and T. Meyarivan. “A Fast and Elitist
    Multiobjective Genetic Algorithm: NSGA-II.” IEEE Transactions on
    Evolutionary Computation 6, no. 2 (April 2002): 182-97.
    https://doi.org/10.1109/4235.996017.

"""
from typing import Tuple

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

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem
from poli_baselines.core.utils.pymoo import (
    DiscreteSequenceMutation,
    _from_array_to_dict,
    _from_dict_to_array,
)


class Discrete_NSGA_II(AbstractSolver):
    """Discrete NSGA-II solver for multi-objective optimization problems.

    NSGA-II is a multi-objective genetic algorithm that uses a non-dominated
    sort to rank individuals and a crowding distance to maintain diversity [1].

    This discrete NSGA-II solver uses the implementation of NSGA-II inside
    pymoo, and our wrapper for discrete problems s.t. they can be used as
    pymoo Problems. For those acquainted with pymoo: we use Choice variables
    inside our definition of the problem. These Choice variables select letters
    from the alphabet. See the documentation of DiscretePymooProblem for more
    details.

    Parameters:
    -----------
    black_box : AbstractBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        The initial population of solutions.
    y0 : np.ndarray
        The corresponding objective values for the initial population.
    population_size : int, optional
        The size of the population, by default 10.
    initialize_with_x0 : bool, optional
        Whether to initialize the algorithm with the provided x0, by default True.
    mutation : Mutation, optional
        The mutation operator to be used, by default None.
    sampling : Sampling, optional
        The sampling operator to be used, by default None.
    mating : InfillCriterion, optional
        The mating operator to be used, by default None.

    Attributes:
    -----------
    pymoo_problem : DiscretePymooProblem
        The problem instance for pymoo optimization.
    algorithm : NSGA2
        The NSGA-II algorithm instance.

    Methods:
    --------
    next_candidate() -> np.ndarray:
        Get the next candidate solution from the algorithm.

    step() -> Tuple[np.ndarray, np.ndarray]:
        Perform a single step of the algorithm and return the current population and objective values.

    References:
    -----------
    [1] Deb, K., A. Pratap, S. Agarwal, and T. Meyarivan. “A Fast and Elitist
        Multiobjective Genetic Algorithm: NSGA-II.” IEEE Transactions on
        Evolutionary Computation 6, no. 2 (April 2002): 182-97.
        https://doi.org/10.1109/4235.996017.
    """

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
        """
        Initialize the NSGA_II solver.

        Parameters
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial population of solutions.
        y0 : np.ndarray
            The corresponding objective values of the initial population.
        population_size : int, optional
            The size of the population, by default 10.
        initialize_with_x0 : bool, optional
            Whether to initialize the algorithm with the provided x0, by default True.
        mutation : Mutation, optional
            The mutation operator. If None, we use a DiscreteSequenceMutation (see our
            implementation in the API reference).
        sampling : Sampling, optional
            The sampling operator. If None, we use a MixedVariableSampling (see pymoo
            for more details)
        mating : InfillCriterion, optional
            The mating operator. If None, we use a MixedVariableMating with
            MixedVariableDuplicateElimination (see pymoo for more details).
        """
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

        if mutation is None:
            mutation = DiscreteSequenceMutation()

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
        """
        Generate the next candidate solution using the NSGA-II algorithm.

        Returns:
        --------
            x: np.ndarray: The current population of solutions.
        """
        x = self.algorithm.pop.get("X")
        return x

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single step of the algorithm, returning population and objective values.

        This implementation uses the ask-tell interface of pymoo, as described in the
        documentation [1].

        Returns:
        --------
            x: np.ndarray: The current population of solutions.
            y: np.ndarray: The current objective values.

        References:
        -----------
        [1] https://pymoo.org/algorithms/usage.html#Problem-Depdendent
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
        x_as_array = _from_dict_to_array(x)
        self.update(x_as_array, y)
        self.iteration += 1

        return x, y
