from typing import Tuple
from itertools import product

import torch
import numpy as np


def _image_from_values(
    values: torch.Tensor,
    limits: Tuple[float, float],
    n_points_in_grid: int,
):
    """
    Transforms a tensor of values into an
    {n_points_in_grid}x{n_points_in_grid} image.
    """
    z1s = torch.linspace(*limits, n_points_in_grid)
    z2s = torch.linspace(*limits, n_points_in_grid)

    fine_grid = torch.Tensor([[x, y] for x, y in product(z1s, z2s)])
    p_dict = {(x.item(), y.item()): v.item() for (x, y), v in zip(fine_grid, values)}

    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1s)
        for i, y in enumerate(reversed(z2s))
    }

    p_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        p_img[i, j] = p_dict[z]

    return p_img
