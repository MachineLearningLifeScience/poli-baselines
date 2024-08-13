"""
This module contains ways of transforming poli's objective
functions to pymoo problems.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path
import pickle

import numpy as np

from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.variable import Choice
from pymoo.core.population import Population

from poli.core.multi_objective_black_box import MultiObjectiveBlackBox


def _from_dict_to_array(x: List[dict]) -> np.ndarray:
    """
    Transforms a list of dicts {x_i: value}
    to an array of shape [n, sequence_length]
    """
    if isinstance(x, dict):
        new_x = np.array([x[f"x_{i}"] for i in range(len(x))])
    else:
        new_x = np.array([[x_[f"x_{i}"] for i in range(len(x_))] for x_ in x])

    return new_x


def _from_array_to_dict(x: np.ndarray) -> List[dict]:
    """Transforms an array [[x_0, x_1, ...]_i] to a dict {x_0: value, x_1: value, ...}"""
    dicts = []
    for row in x:
        dict_ = {}
        for i in range(len(row)):
            dict_[f"x_{i}"] = row[i]

        dicts.append(dict_)

    return dicts


class DiscretePymooProblem(Problem):
    """
    A class representing a discrete optimization problem using the Pymoo library.

    Parameters:
    -----------
    black_box : MultiObjectiveBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        The initial population of solutions.
    y0 : np.ndarray
        The corresponding objective values for the initial population.
    checkpoint_path : Path, optional
        The path to the checkpoint file for saving and loading the population.
    initialize_with_x0 : bool, optional
        Whether to initialize the population with the provided x0 and y0.
    **kwargs : dict
        Additional keyword arguments to be passed to the base class.

    Attributes:
    -----------
    black_box : MultiObjectiveBlackBox
        The black box function to be optimized.
    black_box_for_minimization : MultiObjectiveBlackBox
        The black box function for minimization (negative of the original black box function).
    x0 : np.ndarray
        The initial population of solutions.
    y0 : np.ndarray
        The corresponding objective values for the initial population.
    alphabet : List[str]
        The list of possible choices for each variable.
    checkpoint_path : Path or None
        The path to the checkpoint file for saving and loading the population.
    """

    def __init__(
        self,
        black_box: MultiObjectiveBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        checkpoint_path: Path = None,
        initialize_with_x0: bool = True,
        minimize: bool = False,
        alphabet: list[str] = None,
        sequence_length: int | None = None,
        **kwargs,
    ):
        """
        Initializes a DiscretePymooProblem instance.

        Parameters:
        -----------
        black_box : MultiObjectiveBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial population of solutions.
        y0 : np.ndarray
            The corresponding objective values for the initial population.
        checkpoint_path : Path, optional
            The path to the checkpoint file for saving and loading the population.
        initialize_with_x0 : bool, optional
            Whether to initialize the population with the provided x0 and y0.
        **kwargs : dict
            Additional keyword arguments to be passed to the base class.
        """
        self.black_box = black_box
        if minimize:
            self.black_box_for_minimization = black_box
        else:
            self.black_box_for_minimization = -black_box
        self.x0 = x0
        self.y0 = y0

        # We define sequence_length discrete choice variables,
        # selecting from the alphabet, which we assure is List[str]
        # (by checking the first key, if it's dict).
        if alphabet is None:
            alphabet = black_box.info.alphabet

        if isinstance(alphabet, Dict):
            alphabet = list(alphabet.keys())
            assert isinstance(alphabet[0], str)

        self.alphabet = alphabet

        if sequence_length is None:
            sequence_length = x0.shape[1]

        variables = {f"x_{i}": Choice(options=alphabet) for i in range(sequence_length)}

        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            X, F = self._load_checkpoint()
            pop = Population.new("X", X)
            pop.set("F", F)
        else:
            if initialize_with_x0:
                pop = Population.new("X", x0)
                if minimize:
                    pop.set("F", y0)
                else:
                    pop.set("F", -y0)
            else:
                pop = None

        super().__init__(
            vars=variables,
            n_obj=y0.shape[1],
            sampling=pop,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the black box function by transforming
        the discrete choices to a vector, and then evaluating.
        """
        # At this point, x is a dictionary with keys "x_0", "x_1", etc.
        # and we assume that were evaluating a single x at a time.
        x = _from_dict_to_array(x)

        # The output is a [1, n] array, where n is the number of objectives
        f = self.black_box_for_minimization(x, context=kwargs.get("context", None))
        out["F"] = f

    def save_checkpoint(self, saving_path: Path):
        """
        Saves the current population to a checkpoint file (using pickle).

        Parameters:
        -----------
        saving_path : Path
            The path to the checkpoint file.
        """
        X = self.pop.get("X")
        F = self.pop.get("F")

        with open(saving_path, "wb") as fp:
            pickle.dump((X, F), fp)

    def _load_checkpoint(self) -> Tuple[dict, list]:
        """
        Loads the current population from a checkpoint file (using pickle).

        Returns:
        --------
        X : dict
            The current population.
        F : list
            The corresponding objective values.
        """
        with open(self.checkpoint_path, "rb") as fp:
            X, F = pickle.load(fp)

        return X, F
