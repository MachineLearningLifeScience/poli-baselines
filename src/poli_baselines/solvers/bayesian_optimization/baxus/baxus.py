"""This module implements 'Bayesian Optimization with adaptively expanding subspaces' (BAXUS) [1].

This implementation is based on the tutorial provided inside BoTorch [2].

References
----------
[1] Increasing the scope as you learn: adaptive Bayesian
    Optimization in Nested Subspaces (TODO: complete).
[2] https://botorch.org/tutorials/baxus
"""

from typing import Tuple, Type, Union

import numpy as np

from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from botorch.acquisition import AcquisitionFunction, ExpectedImprovement
from botorch.models import SingleTaskGP

from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.utils.acquisition.thompson_sampling import ThompsonSampling
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)


class BAxUS(BaseBayesianOptimization):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: Type[AcquisitionFunction] = ThompsonSampling,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10,
    ):
        super().__init__(
            black_box=black_box,
            x0=x0,
            y0=y0,
            mean=mean,
            kernel=kernel,
            acq_function=acq_function,
            bounds=bounds,
            penalize_nans_with=penalize_nans_with,
        )

    def _compute_trust_region(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the lower and upper bounds of the trust region."""
        ...

    def _optimize_acquisition_function(
        self, acquisition_function: ThompsonSampling
    ) -> np.ndarray:
        if isinstance(acquisition_function, ThompsonSampling):
            return acquisition_function()
        elif isinstance(acquisition_function, ExpectedImprovement):
            trust_region_bounds = self._compute_trust_region()
            return super()._optimize_acquisition_function(
                acquisition_function, bounds=trust_region_bounds
            )
        else:
            raise NotImplementedError(
                "This method is only implemented for ThompsonSampling or ExpectedImprovement."
            )

    def _instantiate_acquisition_function(
        self, model: SingleTaskGP
    ) -> Union[Type[ThompsonSampling], Type[ExpectedImprovement]]:
        if self.acq_function == ThompsonSampling:
            thomspon_sampling_acq_function = self.acq_function(model=model)
            return thomspon_sampling_acq_function
        elif self.acq_function == ExpectedImprovement:
            return super()._instantiate_acquisition_function(model=model)
        else:
            raise NotImplementedError(
                "This method is only implemented for ThompsonSampling or ExpectedImprovement."
            )

    def _compute_random_embedding_matrix(
        self, input_dim: int, subspace_dim: int
    ) -> np.ndarray:
        """Computes an initial random embedding.

        This random embedding matrix is a {subspace_dim}x{input_dim} matrix.
        Its entries are either 0, 1, or -1, and each column contains exactly one
        non-zero value. See Def. 1. in [1].

        This implementation is based on the function "embedding_matrix" in [2].

        Parameters
        ----------
        input_dim : int
            Dimension of the whole input space.
        target_dim : int
            Dimension of the target subspace.

        Returns
        -------
        S : np.ndarray
            Random embedding matrix.

        References
        ----------
        [1] Increasing the scope as you learn: adaptive Bayesian
            Optimization in Nested Subspaces (TODO: complete).
        [2] https://botorch.org/tutorials/baxus
        """
        if (
            subspace_dim >= input_dim
        ):  # return identity matrix if target size greater than input size
            return np.eye(input_dim)

        # Choosing a random permutation of the input dimensions,
        # and batching them so that each batch has almost equal number of dimensions.
        input_dimensions_permuted = np.random.permutation(input_dim)

        # Split the dimensions into almost equally-sized bins.
        n_contributions_per_lower_dimension = input_dim // subspace_dim
        n_remaining_contributions = input_dim % subspace_dim

        # We construct a dictionary {row_index: [indices]} where each row index
        # corresponds to a row in the embedding matrix, and the list of indices
        # corresponds to the indices of the input dimensions that are non-zero
        # in that row.
        indices_per_row = {}

        # First, we batch the dimensions according to input_dim // subspace_dim.
        for embedding_dimension in range(subspace_dim):
            indices_per_row[embedding_dimension] = input_dimensions_permuted[
                embedding_dimension
                * n_contributions_per_lower_dimension : (embedding_dimension + 1)
                * n_contributions_per_lower_dimension
            ]

        # Finally, we add the remaining dimensions to the last rows. At this point,
        # we know n_remaining_contributions < subspace_dim.
        for embedding_dimension in range(n_remaining_contributions):
            indices_per_row[embedding_dimension] = np.append(
                indices_per_row[embedding_dimension],
                input_dimensions_permuted[-embedding_dimension - 1],
            )

        # Creating the embedding matrix.
        S = np.zeros((subspace_dim, input_dim))
        for embedding_dimension in range(subspace_dim):
            # We assign either +1 or -1 at random sign to each element in the row.
            S[embedding_dimension, indices_per_row[embedding_dimension]] = (
                2
                * np.random.randint(2, size=indices_per_row[embedding_dimension].shape)
                - 1
            )

        return S

    def _expand_embedding_matrix_and_observations(
        self,
        embedding_matrix: np.ndarray,
        current_observations: np.ndarray,
        n_new_bins: int,
    ) -> np.ndarray:
        """Expands the embedding matrix, and the current observations.

        This method expands the embedding matrix by adding a new column to it,
        in such a way that the current observations can still be used.
        See Algorithm 2. in [1].

        This implementation is based on the function
        "increase_embedding_and_observations" in [2].

        In plain English, this algorithm works as follows:
        For each row in the current embedding matrix, do the following:
        1. Identify the "bin", i.e. the non-zero elements and their positions.
           These will be split into new bins, according to {n_new_bins}.
        2. Split the current bin in the row into {n_new_bins} new bins,
           and keep approximately half of them (to encourage balance between
           the number of dimensions in each bin).
        3. Create a new matrix with this latter half of the bin. To do so,
           consider the number of bins in this latter half and, for each of
           them, create a new row by moving the associated non-zero elements
           from the original row to the new row.
        4. Set the non-zero elements in the original row to zero.
        5. For each observation column associated with this row, repeat it as many
           times as the number of bins in the latter half of the original row.

        Returns
        -------
        S : np.ndarray
            Expanded embedding matrix.
        X : np.ndarray
            Expanded observations.

        References
        ----------
        [1] Increasing the scope as you learn: adaptive Bayesian
            Optimization in Nested Subspaces (TODO: complete).
        [2] https://botorch.org/tutorials/baxus
        """
        assert (
            current_observations.shape[1] == embedding_matrix.shape[0]
        ), "Observations don't lie in row space of S"

        embedding_matrix_update = embedding_matrix.copy()
        observations_update = current_observations.copy()

        for k, (row, column) in enumerate(
            zip(embedding_matrix, current_observations.T)
        ):
            # Step 1: Identify the "bin", i.e. the non-zero elements and their positions.
            non_zero_elements = np.nonzero(row)[0]

            # Step 2: Split the current bin in the row into {n_new_bins} new bins,
            # and keep approximately half of them (to encourage balance between
            # the number of dimensions in each bin).
            n_row_bins = min(
                n_new_bins, len(non_zero_elements)
            )  # number of new bins is always less or equal than the contributing input dims in the row minus one

            new_bins = np.array_split(non_zero_elements, n_new_bins)[
                n_new_bins // 2 :
            ]  # the dims in the first bin won't be moved

            # Step 3: Create a new matrix with this latter half of the bin. To do so,
            # consider the number of bins in this latter half and, for each of
            # them, create a new row by moving the associated non-zero elements
            # from the original row to the new row.
            new_submatrix = np.zeros((len(new_bins), len(row)))
            for i, new_bin in enumerate(new_bins):
                new_submatrix[i, new_bin] = row[new_bin]

                # Step 4: Set the non-zero elements in the original row to zero.
                embedding_matrix_update[k, new_bin] = 0

            # Step 5: For each observation column associated with this row, repeat it as many
            # times as the number of bins in the latter half of the original row.
            observations_update = np.hstack(
                (
                    observations_update,
                    column.reshape(-1, 1).repeat(len(new_bins), axis=1),
                )
            )

            # Finally, stack the new submatrix on top of the original embedding matrix.
            embedding_matrix_update = np.vstack(
                (embedding_matrix_update, new_submatrix)
            )

        return embedding_matrix_update, observations_update

    def next_candidate(self) -> np.ndarray:
        ...
