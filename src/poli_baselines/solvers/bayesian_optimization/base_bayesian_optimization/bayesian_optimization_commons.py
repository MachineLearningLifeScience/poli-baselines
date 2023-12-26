"""This is a module of utilities containing common functions for
the Bayesian optimization solvers."""
from typing import Literal, Tuple

import numpy as np
import torch

from botorch.acquisition import AcquisitionFunction


def optimize_acquisition_function_using_grid_search(
    n_dimension: Literal[1, 2],
    acquisition_function: AcquisitionFunction,
    bounds: Tuple[float, float],
    n_points_in_acq_grid: int = 100,
) -> np.ndarray:
    """Optimizes the acquisition function using grid
    search (for dimensions 1 and 2), breaking ties at random
    using numpy.random.choice.

    Parameters
    ----------
    n_dimension : int
        The number of dimensions of the acquisition function.
    acquisition_function : AcquisitionFunction
        The acquisition function to optimize.
    bounds : Tuple[float, float]
        The bounds of the acquisition function.
    n_points_in_acq_grid : int
        The number of points in the grid to search for the maximum.

    Returns
    -------
    candidate : np.ndarray
        The candidate point.
    """
    assert n_dimension in [
        1,
        2,
    ], "The grid search is only implemented for dimensions 1 and 2."
    if n_dimension == 2:
        zs = torch.Tensor(
            [
                [x, y]
                for x in torch.linspace(*bounds, n_points_in_acq_grid)
                for y in reversed(torch.linspace(*bounds, n_points_in_acq_grid))
            ]
        )
        acq_values = acquisition_function(zs.unsqueeze(1))

        # This is the argmax as a set,
        possible_candidates = zs[acq_values == acq_values.max()]

        # and so we select one of these possible candidates at random
        candidate = possible_candidates[
            np.random.choice(len(possible_candidates), 1)[0]
        ].reshape(1, 2)

    elif n_dimension == 1:
        zs = torch.Tensor([[x] for x in torch.linspace(*bounds, n_points_in_acq_grid)])
        acq_values = acquisition_function(zs.unsqueeze(1))

        # This is the argmax as a set,
        possible_candidates = zs[acq_values == acq_values.max()]

        # and so we select one of these possible candidates at random
        candidate = possible_candidates[
            np.random.choice(len(possible_candidates), 1)[0]
        ].reshape(1, 1)

    return candidate.numpy(force=True)
