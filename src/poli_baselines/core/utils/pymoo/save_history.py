"""
This module implements history saving utilities
from pymoo-related objects.
"""

import json
from pathlib import Path
from typing import Dict, List

from pymoo.core.result import Result


def _from_dict_to_list(d: Dict[str, str]):
    """
    Since we are using Choice variables on pymoo, we need to
    convert the dictionary to a list. The dictionary has the
    following format: {"x_0": ..., "x_1": ..., }
    """
    return [d[f"x_{i}"] for i in range(len(d))]


def _from_list_to_dict(list_of_strings: List[str]) -> Dict[str, str]:
    """
    Since we are using Choice variables on pymoo, we need to
    convert the list to a dictionary. The dictionary has the
    following format: {"x_0": ..., "x_1": ..., }
    """
    return {f"x_{i}": list_of_strings[i] for i in range(len(list_of_strings))}


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
