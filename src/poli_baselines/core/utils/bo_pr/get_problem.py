import torch

from botorch.utils.multi_objective import infer_reference_point

from discrete_mixed_bo.problems.base import DiscreteTestProblem

from .poli_objective_in_pr import PoliObjective, PoliMultiObjective


def get_problem(name: str, **kwargs) -> DiscreteTestProblem:
    r"""Initialize the test function."""
    if name == "poli":
        return PoliObjective(
            black_box=kwargs["black_box"],
            alphabet=kwargs.get("alphabet", None),
            sequence_length=kwargs.get("sequence_length", None),
            negate=kwargs.get("negate", False),
        )
    elif name == "poli_moo":
        alphabet = kwargs.get("alphabet", None)
        s_len = kwargs.get("sequence_length", None)
        if s_len is None:
            raise RuntimeError("Sequence Length None!")
        integer_bounds = torch.zeros(2, s_len)
        integer_bounds[1, :] = len(alphabet)
        problem = PoliMultiObjective(
            black_box=kwargs["black_box"],
            alphabet=alphabet,
            sequence_length=kwargs.get("sequence_length", None),
            negate=kwargs.get("negate", False),
            ref_point=infer_reference_point(
                torch.from_numpy(kwargs.get("y0", None))
            ),  # NOTE from infer_reference_point: this assumes maximization of all objectives.
            integer_indices=list(range(s_len)),
            integer_bounds=integer_bounds,
        )
        return problem

    else:
        raise ValueError(f"Unknown function name: {name}!")
