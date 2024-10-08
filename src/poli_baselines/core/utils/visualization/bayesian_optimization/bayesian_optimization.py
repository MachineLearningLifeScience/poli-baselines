from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from botorch.acquisition import AcquisitionFunction
from gpytorch.models import ExactGP

from ..common import _image_from_values


def plot_prediction_in_2d(
    model: ExactGP,
    ax: plt.Axes,
    limits: Tuple[float, float],
    historical_x: torch.Tensor = None,
    candidate: torch.Tensor = None,
    cmap: str = None,
    colorbar_limits: Tuple[float, float] = (None, None),
) -> None:
    """
    Plots mean of the GP in the axes in a fine grid
    in latent space. Assumes that the latent space
    is of size 2.
    """
    n_points_in_grid = 75

    fine_grid_in_latent_space = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(*limits, n_points_in_grid),
                torch.linspace(*limits, n_points_in_grid),
            )
        ]
    )

    predicted_distribution = model(fine_grid_in_latent_space)
    means = predicted_distribution.mean
    means_as_img = _image_from_values(means, limits, n_points_in_grid)

    lower, upper = colorbar_limits

    plot = ax.imshow(
        means_as_img, extent=[*limits, *limits], cmap=cmap, vmin=lower, vmax=upper
    )

    if historical_x is not None:
        ax.scatter(
            historical_x[:, 0], historical_x[:, 1], c="white", edgecolors="black"
        )
        ax.scatter(
            [historical_x[-1, 0]], [historical_x[-1, 1]], c="red", edgecolors="black"
        )

    if candidate is not None:
        ax.scatter([candidate[0]], [candidate[1]], c="blue", edgecolors="black")

    if colorbar_limits != (None, None):
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)


def plot_acquisition_in_2d(
    acq_function: AcquisitionFunction,
    ax: plt.Axes,
    limits: Tuple[float, float],
    historical_x: torch.Tensor = None,
    candidate: torch.Tensor = None,
    cmap: str = "Blues",
    plot_colorbar: bool = False,
):
    n_points_in_grid = 75

    fine_grid_in_latent_space = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(*limits, n_points_in_grid),
                torch.linspace(*limits, n_points_in_grid),
            )
        ]
    ).unsqueeze(1)

    acq_values = acq_function(fine_grid_in_latent_space)
    acq_values_as_img = _image_from_values(acq_values, limits, n_points_in_grid)

    plot = ax.imshow(acq_values_as_img, extent=[*limits, *limits], cmap=cmap)

    if historical_x is not None:
        ax.scatter(
            historical_x[:, 0], historical_x[:, 1], c="white", edgecolors="black"
        )
        ax.scatter(
            [historical_x[-1, 0]], [historical_x[-1, 1]], c="red", edgecolors="black"
        )

    if candidate is not None:
        ax.scatter([candidate[0]], [candidate[1]], c="blue", edgecolors="black")

    if plot_colorbar:
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
