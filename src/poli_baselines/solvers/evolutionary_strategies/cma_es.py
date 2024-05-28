"""CMA-ES poli solver based on pycma [1].

CMA-ES (which stands for Covariance Matrix
Adaptation - Evolutionary Strategy) is an optimization
algorithm which evolves the parameters of a multivariate
normal distribution, and samples from it to generate
candidate solutions [2].

References
----------
[1] pycma: https://github.com/CMA-ES/pycma
[2] N. Hansen and A. Ostermeier, "Adapting arbitrary normal
    mutation distributions in evolution strategies: the covariance
    matrix adaptation," Proceedings of IEEE International Conference
    on Evolutionary Computation, Nagoya, Japan, 1996, pp. 312-317,
    doi: 10.1109/ICEC.1996.542381.

"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

import cma

from poli_baselines.core.step_by_step_solver import StepByStepSolver


class CMA_ES(StepByStepSolver):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) solver.

    Parameters:
    -----------
    black_box : AbstractBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        The initial solution vector.
    y0 : np.ndarray
        The initial objective function values corresponding to x0.
    initial_mean : np.ndarray
        The initial mean of the multivariate normal distribution.
    initial_sigma : float, optional
        The initial step size for the CMA-ES algorithm. Default is 1.0.
    population_size : int, optional
        The population size for the CMA-ES algorithm. Default is 10.

    Attributes:
    -----------
    initial_sigma : float
        The initial step size for the CMA-ES algorithm.
    population_size : int
        The population size for the CMA-ES algorithm.

    Methods:
    --------
    step() -> Tuple[np.ndarray, np.ndarray]
        Runs the solver for one iteration. This
        method overrides the step method from the
        AbstractSolver class.
    next_candidate() -> np.ndarray
        Returns the next candidate solutions.

    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        initial_mean: np.ndarray,
        initial_sigma: float = 1.0,
        population_size: int = 10,
        bounds: tuple[float, float] | None = None,
    ):
        """
        Initialize the CMA-ES solver.

        Parameters
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial solution vector.
        y0 : np.ndarray
            The initial fitness values corresponding to the solution vector.
        initial_sigma : float, optional
            The initial standard deviation for the mutation, by default 1.0.
        population_size : int, optional
            The number of individuals in the population, by default 10.
        """
        super().__init__(black_box, x0, y0)

        self.initial_sigma = initial_sigma
        self.population_size = population_size

        opts = {
            "popsize": population_size,
        }

        if bounds is not None:
            opts["bounds"] = bounds

        self._internal_cma = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, opts)

        # Include the x0 and y0
        _ = self._internal_cma.ask()
        self._internal_cma.tell(x0, -y0)

    def next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solutions.

        Returns:
        --------
        x_population: np.ndarray
            The next candidate solutions as a 2D array
            of size [population_size, n_dimensions]
        """
        arrays = self._internal_cma.ask()

        return np.vstack(arrays)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the solver for one iteration.

        Returns:
        --------
        x: np.ndarray
            The next candidate solutions.
        y: np.ndarray
            The fitness values of the next sample from the population.
        """
        x = self.next_candidate()
        y = self.black_box(x)

        # Since the CMA-ES is expecting a function
        # to minimize, we need to negate the objective
        # function inside the tell method.
        self._internal_cma.tell(x, -y)

        # We update the history
        self.update(x, y)
        self.iteration += 1

        return x, y
