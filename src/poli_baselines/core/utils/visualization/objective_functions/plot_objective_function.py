"""This module implements tools for visualizing 2D objective functions."""
from itertools import product
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from poli.core.abstract_black_box import AbstractBlackBox

from ..common import _image_from_values


def plot_objective_function(
    f: AbstractBlackBox,
    ax: plt.Axes,
    limits: Tuple[float, float],
    cmap: str = None,
    colorbar_limits: Tuple[float, float] = (None, None),
):
    """
    Plots the objective function in a fine grid in latent space.
    """

    n_points_in_grid = 75

    fine_grid_in_latent_space = np.array(
        [
            [x, y]
            for x, y in product(
                np.linspace(*limits, n_points_in_grid),
                np.linspace(*limits, n_points_in_grid),
            )
        ]
    )

    # We call f._black_box instead of f, because we want to
    # avoid the whole logging logic.
    function_values = f._black_box(fine_grid_in_latent_space)

    function_values_as_img = _image_from_values(
        function_values, limits, n_points_in_grid
    )

    lower, upper = colorbar_limits
    plot = ax.imshow(
        function_values_as_img,
        extent=[*limits, *limits],
        cmap=cmap,
        vmin=lower,
        vmax=upper,
    )

    if colorbar_limits != (None, None):
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
