"""
This can run inside the bounce env.
"""

from __future__ import annotations

import numpy as np
import torch
from bounce.benchmarks import SyntheticBenchmark
from bounce.util.benchmark import Parameter, ParameterType
from poli.core.abstract_black_box import AbstractBlackBox
from poli.repository import ToyContinuousBlackBox


class PoliBenchmark(SyntheticBenchmark):
    def __init__(
        self,
        f: AbstractBlackBox,
        noise_std: float,
        *args,
        sequence_length: int | None = None,
        alphabet: list[str] | None = None,
        **kwargs,
    ):
        # Depending on the nature of the problem, build the
        # parameters.
        self.f = f
        if f.info.is_discrete():
            if alphabet is None:
                assert f.info.alphabet is not None or not f.info.is_discrete()
                alphabet = f.info.alphabet
            self.alphabet_i_to_s = {
                index: token for index, token in enumerate(alphabet)
            }
            self.alphabet_s_to_i = {
                token: index for index, token in enumerate(alphabet)
            }
            self.n_realizations = len(alphabet)
        else:
            assert alphabet is None
            self.alphabet_i_to_s = None
            self.alphabet_s_to_i = None
            self.n_realizations = None

        parameters = self.build_parameters(f, sequence_length=sequence_length)
        super().__init__(parameters, noise_std, *args, **kwargs)

    def _call_discrete(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        assert self.f.info.is_discrete()
        start = 0
        _x = []
        for parameter in self.parameters:
            end = start + parameter.dims_required
            one_hot = x[start:end]
            # transform onehot to categorical
            cat = torch.argmax(one_hot)
            token = self.alphabet_i_to_s[int(cat.item())]
            _x.append(token)
            start = end
        _x = np.array(_x)

        # TODO: is Bounce maximizing or minimizing?
        return -self.f(_x.reshape(1, -1)).flatten()

    def _call_continuous(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        # TODO: is Bounce maximizing or minimizing?
        return -self.f(x.reshape(1, -1)).flatten()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        call_function = (
            self._call_discrete if self.f.info.is_discrete() else self._call_continuous
        )
        y_ = [call_function(x_i) for x_i in x]
        y = np.vstack(y_)
        return torch.tensor(y)

    def build_parameters(self, f: AbstractBlackBox, sequence_length: int = None):
        if f.info.is_discrete():
            return self._build_discrete_parameters(f, sequence_length)
        else:
            return self._build_continuous_parameters(f)

    def _build_continuous_parameters(self, f: ToyContinuousBlackBox):
        lower_bound, upper_bound = f.bounds
        return [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            for i in range(f.n_dimensions)
        ]

    def _build_discrete_parameters(
        self, f: AbstractBlackBox, sequence_length: int | None = None
    ):
        if sequence_length is None:
            if not isinstance(f, ToyContinuousBlackBox):
                sequence_length = f.info.max_sequence_length

                assert sequence_length < float("inf")

        return [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CATEGORICAL,
                lower_bound=0,
                upper_bound=self.n_realizations - 1,
            )
            for i in range(sequence_length)
        ]
