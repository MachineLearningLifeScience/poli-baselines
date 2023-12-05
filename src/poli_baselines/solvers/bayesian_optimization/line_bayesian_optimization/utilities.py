"""This module contains utilities used inside line Bayesian Optimization."""

from typing import List, Optional, Tuple
import numpy as np


def ray_box_intersection(
    p: np.ndarray, v: np.ndarray, bounds: List[Tuple[float, float]]
) -> Optional[Tuple[float, np.ndarray]]:
    """
    Find the intersection between a ray and an n-dimensional box.

    Parameters:
        p (np.ndarray): Point inside the box.
        v (np.ndarray): Direction vector of the ray.
        bounds (List[Tuple[float, float]]): List of tuples, each representing the bounds for one dimension (-L, L).

    Returns:
        Optional[Tuple[float, np.ndarray]]: t and intersection point if it exists, (None, None) otherwise.
    """

    n = len(p)  # Dimensionality of the space

    # Initialize the intersection interval
    t_min = float("-inf")
    t_max = float("inf")

    # Check each dimension
    for i in range(n):
        # Check if the ray is parallel to the box in this dimension
        if abs(v[i]) < 1e-6:
            # Ray is parallel, check if the point is within the bounds
            if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
                return None  # No intersection
        else:
            # Compute intersection with the planes of the box in this dimension
            t1 = (bounds[i][0] - p[i]) / v[i]
            t2 = (bounds[i][1] - p[i]) / v[i]

            # Update the intersection interval
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

    # Check if the intersection interval is valid
    if t_min <= t_max:
        # Intersection point
        intersection_point = p + t_min * v
        return t_min, intersection_point
    else:
        return None, None  # No intersection
