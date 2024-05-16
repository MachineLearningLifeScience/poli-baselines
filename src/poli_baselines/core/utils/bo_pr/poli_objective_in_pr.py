"""
Implements a bridge between poli black boxes
and the way in which PR expects objectives.

This module is only meant to be imported while
running inside the poli__pr environment.

(Pattern-matching off of pest.py)
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
from botorch.utils.torch import BufferDict

from poli.core.abstract_black_box import AbstractBlackBox

from discrete_mixed_bo.problems.base import (
    DiscreteTestProblem,
    MultiObjectiveTestProblem,
)


class PoliObjective(DiscreteTestProblem):
    """
    A bridge between poli black boxes and PR.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        sequence_length: int,
        alphabet: list[str] | None = None,
        noise_std: float | None = None,
        negate: bool = False,
    ) -> None:
        self.dim = sequence_length
        self.black_box = black_box
        alphabet = alphabet or self.black_box.info.alphabet
        if alphabet is None:
            raise ValueError("Alphabet must be provided.")

        self._bounds = [(0, len(alphabet) - 1) for _ in range(sequence_length)]
        self.alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(alphabet)}
        super().__init__(noise_std, negate, categorical_indices=list(range(self.dim)))

    def evaluate_true(self, X: torch.Tensor):
        # Evaluate true seems to be expecting
        # a tensor of integers.
        if X.ndim == 1:
            X = X.unsqueeze(0)

        # 1. transform to a list of strings
        x_str = [[self.alphabet_i_to_s[i] for i in x_i] for x_i in X.numpy(force=True)]

        # 2. evaluate the black box
        return torch.from_numpy(self.black_box(np.array(x_str)))


class PoliDiscreteObjective(DiscreteTestProblem):
    """
    A bridge between poli black boxes and PR. Strictly discrete single objective - no one-hot.
    """

    _discrete_values: dict
    _bounds: list

    def __init__(
        self,
        black_box: AbstractBlackBox,
        sequence_length: int,
        alphabet: list[str] | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
        tokenizer: object = None,
    ) -> None:
        self.dim = sequence_length
        self.black_box = black_box
        self.tokenizer = tokenizer
        alphabet = alphabet or self.black_box.info.alphabet
        if alphabet is None:
            raise ValueError("Alphabet must be provided.")
        # if integer_indices is None:
        #     integer_indices = [i for i in range(sequence_length)]

        self._bounds = [(0, len(alphabet) - 1) for _ in range(sequence_length)]
        self.alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(alphabet)}
        super().__init__(
            noise_std, negate, categorical_indices=list(range(sequence_length))
        )
        self._setup(integer_indices=integer_indices, categorical_indices=categorical_indices)
        self.discrete_values = BufferDict()
        self._discrete_values = {
            f"pos_{i}": list(self.alphabet_s_to_i.values())
            for i in range(sequence_length)
        }
        for v in self._discrete_values.values():
            self._bounds.append((0, len(alphabet)))

    def evaluate_true(self, X: torch.Tensor):
        # Evaluate true seems to be expecting
        # a tensor of integers.
        if X.ndim == 1:
            X = X.unsqueeze(0)

        # 1. transform to a list of strings
        x_str = [[self.alphabet_i_to_s[i] for i in x_i] for x_i in X.numpy(force=True)]

        # 2. evaluate the black box
        return torch.from_numpy(self.black_box(np.array(x_str)))


class PoliMultiObjective(DiscreteTestProblem, MultiObjectiveTestProblem):
    """
    A bridge between poli black boxes and PR.
    """

    num_objectives: int
    _ref_point: List[float]
    _discrete_values: dict
    _bounds: list

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        sequence_length: int,
        alphabet: List[str] = None,
        noise_std: float = None,
        negate: bool = False,
        integer_indices=None,
        integer_bounds=None,
        ref_point: List[float] = None,
        preserve_len: bool = True,
    ) -> None:
        self._bounds = [(0, len(alphabet) - 1) for _ in range(sequence_length)]
        if "" == alphabet[0]:
            self._bounds = [
                (1, len(alphabet) - 1) for _ in range(sequence_length)
            ]  # eliminate pad symbol from sampling
        self.dim = sequence_length
        self.black_box = black_box
        alphabet = alphabet or self.black_box.info.alphabet
        self._ref_point = (
            ref_point  #  NOTE: this assumes maximization of all objectives.
        )
        self.num_objectives = ref_point.shape[-1]
        self.sequence_length = sequence_length
        self.Ls = [len(x[x != ""]) for x in x0]
        self.preserve_len = preserve_len
        if alphabet is None:
            raise ValueError("Alphabet must be provided.")

        self.alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(alphabet)}
        MultiObjectiveTestProblem.__init__(
            self,
            noise_std=noise_std,
            negate=negate,
        )
        self._setup(integer_indices=integer_indices)
        self.discrete_values = BufferDict()
        self._discrete_values = {
            f"pos_{i}": list(self.alphabet_s_to_i.values())
            for i in range(sequence_length)
        }
        for v in self._discrete_values.values():
            self._bounds.append((0, len(alphabet)))

    def _consistent_length(self, x: List[str]):
        valid_x = []
        for _x in x:
            cand_len = len(_x[_x != ""])
            if cand_len not in self.Ls:
                closest_len = min(
                    self.Ls, key=lambda x: abs(x - cand_len)
                )  # clip to closest length
                valid_x.append(
                    list(_x[:closest_len]) + [""] * (self.sequence_length - closest_len)
                )
            else:
                valid_x.append(_x)
        return np.vstack(valid_x)

    def evaluate_true(self, X: torch.Tensor):
        # Evaluate true seems to be expecting
        # a tensor of integers.
        if X.ndim == 1:
            X = X.unsqueeze(0)

        # 1. transform to a list of strings
        x_str = [
            [self.alphabet_i_to_s[int(i)] for i in x_i] for x_i in X.numpy(force=True)
        ]
        if self.preserve_len:
            x_str = self._consistent_length(x_str)
        # 2. evaluate the black box
        return torch.from_numpy(self.black_box(np.array(x_str)))
