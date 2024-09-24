"""Tests for CoRel [1]

[1] TODO: add ArXiv ref.
"""

from pathlib import Path
import json

import warnings
import numpy as np

import pytest

TEST_FILES_PATH = Path(__file__).parent.parent.parent / "test_files"


warnings.filterwarnings("ignore")


def load_alphabet() -> list[str]:
    with open(TEST_FILES_PATH / "zinc250k_alphabet_stoi.json") as f:
        alphabet = json.load(f)

    return list(alphabet.keys())


def load_sequence_length() -> int:
    with open(TEST_FILES_PATH / "zinc250k_metadata.json") as f:
        metadata = json.load(f)

    return metadata["max_sequence_length"]


@pytest.mark.slow()
def test_corel_runs():
    """Tests that CoRel instantiates and runs."""
    from poli import objective_factory

    pytest.importorskip("corel")  # We check if we have corel installed
    from poli_baselines.solvers.bayesian_optimization.corel import (
        CorelSolver,
    )

    alphabet = load_alphabet()
    sequence_length = load_sequence_length()

    problem = objective_factory.create(
        name="rdkit_qed", string_representation="SELFIES"
    )
    black_box = problem.black_box

    solver = CorelSolver(
        black_box=black_box,
        alphabet=alphabet,
        sequence_length=sequence_length,
        n_initial_points=10,
    )

    assert solver is not None

    solver.solve(max_iter=1)


if __name__ == "__main__":
    test_corel_runs()
