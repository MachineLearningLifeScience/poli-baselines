"""
This module implements a simple baseline for
black box optimization: performing mutations
at random.

Since the input is discrete the mutations
are performed by randomly selecting a
position in the input and randomly selecting
a new value for that position.
"""
from typing import Dict

import numpy as np

from poli_baselines.core.abstract_solver import AbstractSolver
from poli.core.abstract_black_box import AbstractBlackBox


class RandomMutation(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
    ):
        super().__init__(black_box, x0, y0)
        self.alphabet = black_box.info.alphabet
        self.string_to_idx = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self.alphabet_size = len(self.alphabet)
        self.idx_to_string = {v: k for k, v in self.string_to_idx.items()}

    def next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history.

        In this case, the RandomMutation solver
        simply returns a random mutation of the
        best performing solution so far.
        """
        # Get the best performing solution so far
        best_x = self.history["x"][np.argmax(self.history["y"])]

        # Perform a random mutation
        # TODO: this assumes that x has shape [1, L],
        # what happens with batches? So far, POLi is
        # implemented without batching in mind.
        next_x = best_x.copy()
        pos = np.random.randint(0, len(next_x.flatten()))

        if next_x.dtype.kind in ("i", "f"):
            mutant = np.random.randint(0, self.alphabet_size)
        elif next_x.dtype.kind in ("U", "S"):
            mutant = np.random.choice(list(self.string_to_idx.keys()))
        else:
            raise ValueError(
                f"Unknown dtype for the input: {next_x.dtype}. "
                "Only integer, float and unicode dtypes are supported."
            )

        next_x[0][pos] = mutant

        return next_x
