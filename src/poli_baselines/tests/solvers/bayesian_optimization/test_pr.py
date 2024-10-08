"""Tests for our bridge with Probabilistic Reparametrization [1]

TODO: add reference
"""

import json
import warnings
from pathlib import Path

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
def test_pr_runs():
    """Tests that Probabilistic Reparametrization instantiates and runs."""
    from poli import objective_factory

    pytest.importorskip("discrete_mixed_bo")  # We check if we have PR installed
    from poli_baselines.solvers.bayesian_optimization.pr import (
        ProbabilisticReparametrizationSolver,
    )

    alphabet = load_alphabet()
    sequence_length = load_sequence_length()

    problem = objective_factory.create(
        name="rdkit_qed", string_representation="SELFIES"
    )
    black_box, x0 = problem.black_box, problem.x0
    y0 = black_box(x0)

    x0_ = np.array([["[nop]"] * sequence_length])
    x0_[0, : x0.shape[1]] = x0

    solver = ProbabilisticReparametrizationSolver(
        black_box=black_box,
        x0=x0_,
        y0=y0,
        alphabet=alphabet,
        sequence_length=sequence_length,
    )

    assert solver is not None

    solver.solve(max_iter=1)


if __name__ == "__main__":
    test_pr_runs()
