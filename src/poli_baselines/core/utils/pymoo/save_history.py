"""
This module implements history saving utilities
from pymoo-related objects.
"""
from typing import Dict
from pathlib import Path
import json

import numpy as np

from pymoo.core.result import Result

from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem


def save_final_population(result: Result, alphabet: Dict[str, int], path: Path):
    """
    The format in which we save the history is as follows:
    {
        "x": [x_0, x_1, ..., x_n],
        "y": [y_0, y_1, ..., y_n],
        "alphabet": {
            ...
        },
    }
    """
    history = {
        "x": [x for x in result.X.tolist()],
        "y": [y for y in result.F.tolist()],
        "alphabet": alphabet,
    }

    with open(path, "w") as f:
        json.dump(history, f)


def save_all_populations(result: Result, alphabet: Dict[str, int], path: Path):
    """
    We save all the different populations in the history of
    the optimization.
    """
    history = {
        i: {
            "x": [x for x in history_i.pop.get("X").tolist()],
            "y": [y for y in history_i.pop.get("F").tolist()],
        }
        for i, history_i in enumerate(result.history)
    }
    history["alphabet"] = alphabet

    with open(path, "w") as f:
        json.dump(history, f)

    ...
