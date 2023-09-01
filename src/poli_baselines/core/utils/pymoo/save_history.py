"""
This module implements history saving utilities
from pymoo-related objects.
"""
from typing import Dict
from pathlib import Path
import json

from pymoo.core.result import Result


def _from_dict_to_list(d: Dict[str, str]):
    """
    Since we are using Choice variables on pymoo, we need to
    convert the dictionary to a list. The dictionary has the
    following format: {"x_0": ..., "x_1": ..., }
    """
    return [d[f"x_{i}"] for i in range(len(d))]


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
        "x": [_from_dict_to_list(x) for x in result.X.tolist],
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
            "x": [_from_dict_to_list(x) for x in history_i.pop.get("X").tolist()],
            "y": [y for y in history_i.pop.get("F").tolist()],
        }
        for i, history_i in enumerate(result.history)
    }
    history["alphabet"] = alphabet

    with open(path, "w") as f:
        json.dump(history, f)
