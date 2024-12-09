"""
Tests for the BOSS implementation

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
def test_boss_runs():
    """
    Test BOSS instantiates and runs.
    """
    from poli import objective_factory

    pytest.importorskip("boss")
    from poli_baselines.solvers.bayesian_optimization.boss import BossSolver

    alphabet = load_alphabet()
    sequence_length = load_sequence_length()

    problem = objective_factory.create(
        name="rdkit_qed", string_representation="SMILES"
    )
    black_box = problem.black_box
    x0 = problem.x0
    y0 = black_box(x0)

    solver = BossSolver(
        black_box=black_box,
        x0=x0,
        y0=y0,
        n_initial_points=1,
    )

    assert solver is not None

    solver.solve(max_iter=1)