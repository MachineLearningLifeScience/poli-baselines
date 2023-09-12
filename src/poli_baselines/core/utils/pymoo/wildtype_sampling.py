from typing import List

import numpy as np

from pymoo.core.sampling import Sampling


class WildtypeMutationSampling(Sampling):
    """
    TODO: document
    """

    def __init__(
        self, x_0: np.ndarray, alphabet: List[str], num_mutations: int = 1
    ) -> None:
        """
        TODO: document
        """
        # We assume a shape [b, L], where some of the rows
        # have been padded with empty strings.
        self.x_0 = x_0
        self.num_mutations = num_mutations
        self.alphabet = alphabet
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        """
        This mutation takes a random wildtype (i.e. a row
        of self.x_0), and mutates it at a random place
        self.num_mutations times.

        To comply with what pymoo expects for discrete
        (choice) problems, This needs to output a list
        of dictionaries, where each dictionary has keys
        "x_0", "x_1", etc. and values that are elements
        of the alphabet.
        """
        # Defining the mutations
        mutations = []
        for _ in range(n_samples):
            # Selecting a row from self.x_0 at random
            random_row = np.random.randint(0, self.x_0.shape[0])
            random_wildtype = self.x_0[random_row, :]

            # Where does the padding start? This will be useful
            # for defining the mutations up until the right index:
            positions_of_padding = np.where(random_wildtype == "")[0]
            if len(positions_of_padding) == 0:
                start_of_padding_idx = random_wildtype.shape[0]
            else:
                start_of_padding_idx = np.where(random_wildtype == "")[0][0]

            sequence_length = random_wildtype.shape[0]

            # The original dict
            mutation = {f"x_{i}": random_wildtype[i] for i in range(sequence_length)}
            all_indices_at_random = np.random.permutation(
                sequence_length - start_of_padding_idx
            )
            indices_to_mutate = all_indices_at_random[: self.num_mutations]

            # Then we mutate the selected indices.
            # (making sure that we don't mutate to the same)
            for idx in indices_to_mutate:
                mutant = np.random.choice(self.alphabet)
                while mutant == mutation[f"x_{idx}"]:
                    mutant = np.random.choice(self.alphabet)

                mutation[f"x_{idx}"] = np.random.choice(self.alphabet)

            # And we add it to the list of mutations
            mutations.append(mutation)

        return mutations
