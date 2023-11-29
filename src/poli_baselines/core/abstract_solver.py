from typing import List, Tuple, Dict, Any
from pathlib import Path
import json

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox


class AbstractSolver:
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
    ):
        self.black_box = black_box
        self.x0 = x0
        self.y0 = y0

        self.history = {
            "x": [x0],
            "y": [y0],
        }

    def next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history.

        TODO: add batch support.
        """
        raise NotImplementedError(
            "This method is abstract, and should be implemented by a subclass."
        )

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Updates the history with the given
        candidate solution and its evaluation.
        """
        # TODO: assert shapes.
        self.history["x"].append(x)
        self.history["y"].append(y)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the solver for one iteration.
        """
        x = self.next_candidate()
        y = self.black_box(x)

        self.update(x, y)

        return x, y

    def solve(
        self,
        max_iter: int = 100,
        break_at_performance: float = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Runs the solver for the given number of iterations.
        :param max_iter:
        :type max_iter:
        :return:
        :rtype:
        """
        # TODO: add logging, link it to the observer logic.
        # TODO: should we add a callback?
        # TODO: should we add a progress bar?
        # TODO: should we add a try-except block?
        for i in range(max_iter):
            _, y = self.step()

            if verbose:
                print(
                    f"Iteration {i}: {y}, best so far: {np.max([y_i for y_i in self.history['y'] if not np.isnan(y_i)])}"
                )

            if break_at_performance is not None:
                if y >= break_at_performance:
                    break

    def save_history(
        self, path: Path, alphabet: List[str] = None, metadata: Dict[str, Any] = None
    ) -> None:
        """
        Saves the history of the solver to the given path.
        """
        x_to_save = [x.flatten().tolist() for x in self.history["x"]]
        y_to_save = [float(y.flatten()[0]) for y in self.history["y"]]

        with open(path, "w") as fp:
            json.dump(
                {"x": x_to_save, "y": y_to_save, "metadata": metadata},
                fp,
            )

    def get_best_solution(self) -> np.ndarray:
        """
        Returns the best solution found so far.
        """
        return self.history["x"][np.nanargmax(self.history["y"])]

    def get_best_performance(self) -> np.ndarray:
        """
        Returns the best performance found so far.
        """
        return np.nanmax(self.history["y"])
