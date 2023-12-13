"""CMA-ES poli solver based on pycma [1].

CMA-ES (which stands for Covariance Matrix
Adaptation - Evolutionary Strategy) is an optimization
algorithm which evolves the parameters of a multivariate
normal distribution, and samples from it to generate
candidate solutions [2].

References
----------
[1] pycma: https://github.com/CMA-ES/pycma
[2] Hansen, N., and A. Ostermeier. “Completely Derandomized
    Self-Adaptation in Evolution Strategies.” Evolutionary
    Computation 9, no. 2 (2001): 159-95.
    https://doi.org/10.1162/106365601750190398.

"""
import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

import cma

from poli_baselines.core.abstract_solver import AbstractSolver


class CMA_ES(AbstractSolver):
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
    next_candidate() -> np.ndarray
        Returns the next candidate solutions.

    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        initial_sigma: float = 1.0,
        population_size: int = 10,
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

        self._internal_cma = cma.CMAEvolutionStrategy(
            x0,
            initial_sigma,
            {
                "popsize": population_size,
            },
        )

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
