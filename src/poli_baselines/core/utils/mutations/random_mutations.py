from typing import List

import numpy as np


def add_random_mutations_to_reach_pop_size(
    x: np.ndarray,
    alphabet: List[str],
    population_size: int,
    unmutable_tokens: list[str] = [""],
):
    """Adds random mutations to x until it reaches population_size.
    # TODO: add n_mutations
    # TODO: this function assumes that the characters in the alphabet
    # are of length 1 (excluding e.g. SELFIES mutations). Fix this.

    Given an array x of shape [b,] or [b, L] (where
    b is less than population_size), this function
    adds random mutations to it until it reaches
    population_size.
    """
    b = x.shape[0]
    if b > population_size:
        raise ValueError(
            f"Expected x to have less elements than the population size. x.shape: {x.shape}, pop_size: {population_size}"
        )

    if len(x.shape) == 1:
        # Then the input as an array [b,] of strings
        x_as_1d_array_of_strings = x
    elif len(x.shape) == 2:
        # Then the input as an array [b, L] of strings
        x_as_1d_array_of_strings = np.array(["".join(x[i, :]) for i in range(b)])
    else:
        raise ValueError(f"Expected x to have shape [b,] or [b, L], but got {x.shape}.")

    more_mutants = []
    for _ in range(population_size - b):
        randomly_selected_string = np.random.choice(x_as_1d_array_of_strings)
        mutant = list(randomly_selected_string)
        random_index = np.random.randint(len(mutant))
        while mutant[random_index] in unmutable_tokens:
            random_index = np.random.randint(len(mutant))
        mutant[random_index] = np.random.choice(alphabet)
        more_mutants.append("".join(mutant))

    if len(x.shape) == 1:
        # Then the input as an array [b,] of strings
        return np.concatenate((x_as_1d_array_of_strings, np.array(more_mutants)))
    elif len(x.shape) == 2:
        # Then we had an input of shape [b, L], so we might
        # have to add padding to the new mutations.
        _, L = x.shape
        return np.concatenate(
            (
                x,
                np.array(
                    [
                        np.array(list(mutation) + [""] * (L - len(mutation)))
                        for mutation in more_mutants
                    ]
                ),
            )
        )
    else:
        raise ValueError(f"Expected x to have shape [b,] or [b, L], but got {x.shape}.")
