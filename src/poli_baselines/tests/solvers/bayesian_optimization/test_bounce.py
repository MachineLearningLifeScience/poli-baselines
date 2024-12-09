"""Tests for our bridge with Bounce

TODO: add reference
"""

import json
import warnings
from pathlib import Path

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
def test_bounce_runs():
    """Tests that Bounce instantiates and runs."""
    from poli import objective_factory

    pytest.importorskip("bounce")  # We check if we have Bounce installed
    from poli_baselines.solvers.bayesian_optimization.bounce import BounceSolver

    alphabet = load_alphabet()
    sequence_length = load_sequence_length()

    problem = objective_factory.create(
        name="rdkit_qed", string_representation="SELFIES"
    )
    black_box = problem.black_box

    solver = BounceSolver(
        black_box=black_box,
        alphabet=alphabet,
        sequence_length=sequence_length,
        n_initial_points=10,
    )

    assert solver is not None

    solver.solve(max_iter=1)


if __name__ == "__main__":
    test_bounce_runs()
