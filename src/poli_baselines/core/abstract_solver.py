from typing import List, Tuple, Dict, Any, Iterable, Callable
from typing_extensions import Self
from pathlib import Path
import json

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from .utils.saving.json_encoders import NumpyToListEncoder


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
            "x": [x0_i.reshape(1, -1) for x0_i in x0],
            "y": [y0_i.reshape(1, -1) for y0_i in y0],
        }

        self.iteration = 0

    def next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history.

        TODO: add batch support.
        """
        raise NotImplementedError(
            "This method is abstract, and should be implemented by a subclass."
        )

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        This method is called after the history is updated.
        """
        pass

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Updates the history with the given
        candidate solution and its evaluation.
        """
        # TODO: assert shapes.
        self.history["x"] += [x_i.reshape(1, -1) for x_i in x]
        self.history["y"] += [y_i.reshape(1, -1) for y_i in y]

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the solver for one iteration.
        """
        x = self.next_candidate()
        y = self.black_box(x)

        self.update(x, y)
        self.post_update(x, y)
        self.iteration += 1

        return x, y

    def solve(
        self,
        max_iter: int = 100,
        break_at_performance: float = None,
        verbose: bool = False,
        pre_step_callbacks: Iterable[Callable[[Self], None]] = None,
        post_step_callbacks: Iterable[Callable[[Self], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the problem for a given number of iterations.

        Parameters
        ----------
        max_iter: int, optional
            The maximum number of iterations to run. By default, 100.
        break_at_performance: float, optional
            If given, the algorithm will stop when the best performance
            is greater than or equal to this value. By default, we don't
            break (i.e. None).
        verbose: bool, optional
            If True, prints the current iteration and the best performance
            found so far. By default, False.
        pre_step_callbacks: Iterable[Callable[[Self], None]], optional
            A list of callbacks to be called before each step. By default,
            None. These callbacks are simply functions that take the solver as
            an argument, and don't return anything. Pre-step callbacks are
            called before self.step() (i.e. before self.next_candidate()).
        post_step_callbacks: Iterable[Callable[[Self], None]], optional
            A list of callbacks to be called after each step. By default,
            None. These callbacks are simply functions that take the solver as
            an argument, and don't return anything. Post-step callbacks are
            called after self.step() (i.e. after self.next_candidate()).

        Returns
        -------
        x: np.ndarray
            The final best-performing solution.
        y: np.ndarray
            The performance of the best-performing solution.
        """
        # TODO: should we add a progress bar?
        # TODO: should we add a try-except block?
        for i in range(max_iter):
            # Call the pre-step callbacks
            if pre_step_callbacks is not None:
                for callback in pre_step_callbacks:
                    callback(self)

            # Take a step, which in turn updates the local history.
            _, y = self.step()

            # Call the post-step callbacks
            if post_step_callbacks is not None:
                for callback in post_step_callbacks:
                    callback(self)

            if verbose:
                print(f"Iteration {i}: {y}, best so far: {self.get_best_performance()}")

            if break_at_performance is not None:
                if y >= break_at_performance:
                    break

        return self.get_best_solution(), self.get_best_performance()

    def save_history(
        self, path: Path, alphabet: List[str] = None, metadata: Dict[str, Any] = None
    ) -> None:
        """
        Saves the history of the solver to the given path.
        """
        x_to_save = [x.flatten().tolist() for x in self.history["x"]]
        y_to_save = [
            float(y.flatten()[0]) if y.shape[1] == 1 else y.tolist()
            for y in self.history["y"]
        ]

        with open(path, "w") as fp:
            json.dump(
                {"x": x_to_save, "y": y_to_save, "metadata": metadata},
                fp,
                cls=NumpyToListEncoder,
            )

    def get_best_solution(self, top_k: int = 1) -> np.ndarray:
        """
        Returns the best solution found so far (assuming that the output is a scalar).

        Returns:
        --------
        one_best_solution: np.ndarray
            The best solution (or one of the best solutions)
            found so far.
        """
        inputs = [x for x in self.history["x"]]
        outputs = [y for y in self.history["y"]]

        stacked_inputs = np.vstack(inputs)
        stacked_outputs = np.vstack(outputs)

        n_objectives = stacked_outputs.shape[1]
        if n_objectives == 1:
            # If we can only return less than top_k solutions,
            # we return the ones we have.
            _top_k = min(top_k, stacked_outputs.shape[0])
            nanargmax = np.argsort(stacked_outputs.flatten())
            best_solutions = stacked_inputs[nanargmax[-_top_k:]]
        else:
            pareto_front = self.get_pareto_front()
            _top_k = min(top_k, pareto_front.shape[0])

            best_solutions = pareto_front[:_top_k]

        return best_solutions.reshape(_top_k, -1)

    def get_best_performance(self, until: int = None) -> np.ndarray:
        """
        Returns the best performance found so far (assuming that the output is a scalar).
        """
        outputs = [y for y in self.history["y"]]

        if until is not None:
            outputs = outputs[:until]

        stacked_outputs = np.vstack(outputs)

        return np.nanmax(stacked_outputs, axis=0)

    def get_pareto_front(self) -> np.ndarray:
        import torch
        from botorch.utils.multi_objective.pareto import is_non_dominated

        x, y = self.get_history_as_arrays()

        assert y.shape[1] >= 2, "This method only works for multi-objective problems."

        y = torch.tensor(y)
        pareto_mask = is_non_dominated(y)
        pareto_front = x[pareto_mask.numpy(force=True)]

        return pareto_front

    def get_history_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the history of the solver as a tuple (x, y).
        """
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)

        return x, y
