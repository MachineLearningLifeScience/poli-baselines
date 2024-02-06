from collections import defaultdict

import numpy as np

from pymoo.core.selection import Selection


class RandomSelectionOfSameLength(Selection):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, _, pop, n_select, n_parents, **kwargs):
        # TODO: implement this by sub-selecting the parents that have a certain length, and forming pairs thereof.

        # This dict will be of the form {length: [index_of_pop_element]},
        # separating the elements of the population according
        # to their length
        subpopulations_of_same_length = defaultdict(list)
        for i, individual in enumerate(pop):
            individuals_length = len([v for v in individual.x.values() if v != ""])
            subpopulations_of_same_length[individuals_length].append(i)

        # For each n_select, we randomly choose a length and select two
        # random parents from said subset.
        # TODO: What should we do if there's only one element of a certain length?
        parents_ = []
        for _ in range(n_select):
            random_length = np.random.choice(list(subpopulations_of_same_length.keys()))
            index_1, index_2 = np.random.choice(
                subpopulations_of_same_length[random_length],
                size=2,
            )
            parents_.append((index_1, index_2))

        return np.array(parents_, dtype=int)
