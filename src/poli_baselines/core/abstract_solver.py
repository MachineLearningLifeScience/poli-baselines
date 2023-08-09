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

    def solve(self, max_iter: int = 1000) -> np.ndarray:
        """
        Runs the solver for the given number of iterations.
        :param max_iter:
        :type max_iter:
        :return:
        :rtype:
        """
        # TODO: add logging, link it to the observer logic.
        for i in range(max_iter):
            x = self.next_candidate()
            y = self.black_box(x)

            self.history["x"].append(x)
            self.history["y"].append(y)

            print(f"Iteration {i}: {y}")
