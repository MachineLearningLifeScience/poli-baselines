"""
This module implements a simple baseline for
black box optimization: performing mutations
at random.

Since the input is discrete the mutations
are performed by randomly selecting a
position in the input and randomly selecting
a new value for that position.
"""

import random

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.step_by_step_solver import StepByStepSolver


class RandomMutation(StepByStepSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        n_mutations: int = 1,
        top_k: int = 1,
        batch_size: int = 1,
        greedy: bool = True,
        alphabet: list[str] | None = None,
    ):
        if x0.ndim == 1:
            x0_ = [list(x_i) for x_i in x0]
            x0 = np.array(x0_)

        super().__init__(black_box, x0, y0)
        self.alphabet = black_box.info.alphabet if alphabet is None else alphabet
        self.alphabet_without_empty = [s for s in self.alphabet if s != ""]
        self.string_to_idx = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self.alphabet_size = len(self.alphabet)
        self.idx_to_string = {v: k for k, v in self.string_to_idx.items() if k != ""}
        self.n_mutations = n_mutations
        self.top_k = top_k
        self.greedy = greedy
        self.batch_size = batch_size

    def _next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history.

        In this case, the RandomMutation solver
        simply returns a random mutation of the
        best performing solution so far.
        """
        if self.greedy:
            # Get the best performing solution so far
            best_xs = self.get_best_solution(top_k=self.top_k)

            # Select one of the best solutions at random
            best_x = random.choice(best_xs)
        else:
            xs, _ = self.get_history_as_arrays()

            random_indices = np.random.permutation(len(xs))
            best_xs = xs[random_indices[: self.top_k]]

        # Perform a random mutation
        # TODO: this assumes that x has shape [1, L],
        # what happens with batches? So far, POLi is
        # implemented without batching in mind.
        next_x = best_x.copy().reshape(1, -1)

        for _ in range(self.n_mutations):
            pos = np.random.randint(0, len(next_x.flatten()))
            while next_x[0][pos] == "":
                pos = np.random.randint(0, len(next_x.flatten()))

            if next_x.dtype.kind in ("i", "f"):
                mutant = np.random.randint(0, self.alphabet_size)
            elif next_x.dtype.kind in ("U", "S"):
                mutant = np.random.choice(self.alphabet_without_empty)
            else:
                raise ValueError(
                    f"Unknown dtype for the input: {next_x.dtype}. "
                    "Only integer, float and unicode dtypes are supported."
                )

            next_x[0][pos] = mutant

        return next_x

    def next_candidate(self) -> np.ndarray:
        mutations = [
            self._next_candidate().reshape(1, -1) for _ in range(self.batch_size)
        ]

        return np.concatenate(mutations, axis=0)
