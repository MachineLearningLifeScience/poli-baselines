"""This module implements 'Bayesian Optimization with adaptively expanding subspaces' (BAxUS).

BAxUS (Bayesian Optimization with adaptively expanding subspaces) [1] is a
Bayesian optimization algorithm that considers a sequence of nested subspaces
of the input space, and adaptively expands the subspace dimension as the
algorithm progresses. The algorithm starts with a random embedding matrix,
and updates it (alongside the observations) as the algorithm progresses.

This implementation is based on the tutorial provided inside BoTorch [2].

References
----------
[1] Papenmeier, Leonard, Luigi Nardi, and Matthias Poloczek. “Increasing the
    Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces,”
    2022. https://openreview.net/forum?id=e4Wf6112DI.
[2] https://botorch.org/tutorials/baxus
"""

from typing import Tuple, Type, Union

import torch
import numpy as np

import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.acquisition import AcquisitionFunction, ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError

from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.utils.acquisition.thompson_sampling import ThompsonSampling
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)

MAX_CHOLESKY_SIZE = float("inf")


class BAxUS(BaseBayesianOptimization):
    """Implements 'Bayesian Optimization with adaptively expanding subspaces' (BAxUS).

    BAxUS (Bayesian Optimization in Adaptive Subspaces) [1] is a class
    that implements the BAxUS algorithm for Bayesian optimization in
    nested subspaces.

    This implementation is based on the tutorial provided inside BoTorch [2].

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        The initial input points for the optimization.
    y0 : np.ndarray
        The corresponding function values for the initial input points.
    mean : Mean, optional
        The mean function of the Gaussian process model. Default is None.
    kernel : Kernel, optional
        The kernel function of the Gaussian process model. Default is None.
    acq_function : Type[AcquisitionFunction], optional
        The type of acquisition function to be used. Default is ThompsonSampling.
    bounds : Tuple[float, float], optional
        The lower and upper bounds of the input space. Default is (-2.0, 2.0).
    penalize_nans_with : float, optional
        The value to penalize NaN function values with. Default is -10.
    initial_subspace_dimension : int, optional
        The initial dimension of the subspace. Default is 2.
    n_new_bins_per_dimension : int, optional
        The number of new bins to split each dimension into. Default is 3.
    initial_trust_region_length : float, optional
        The initial length of the trust region. Default is 0.8.
    success_tolerance : int, optional
        The tolerance for the success counter. Default is 3.
    failure_tolerance : int, optional
        The tolerance for the failure counter. Default is 3.
    trust_region_min_length : float, optional
        The minimum length of the trust region. Default is 0.5**7.
    trust_region_max_length : float, optional
        The maximum length of the trust region. Default is 1.6.
    n_candidates_for_acquisition : int, optional
        The number of candidates to be used for the acquisition function. Default is 100.

    Attributes
    ----------
    gp_model : SingleTaskGP
        The Gaussian process model.
    initial_trust_region_length : float
        The initial length of the trust region.
    trust_region_length : float
        The current length of the trust region.
    trust_region_min_length : float
        The minimum length of the trust region.
    trust_region_max_length : float
        The maximum length of the trust region.
    input_dimension : int
        The dimension of the input space.
    subspace_dimension : int
        The current dimension of the subspace.
    n_new_bins_per_dimension : int
        The number of new bins to split each dimension into.
    embedding_matrix : np.ndarray
        The embedding matrix that maps the input space to the subspace.
    history : dict
        The history of observations in the subspace.
    original_observations : list
        The observations in the original input space.
    success_counter : int
        The counter for successful iterations.
    failure_counter : int
        The counter for failed iterations.
    success_tolerance : int
        The tolerance for the success counter.
    failure_tolerance : int
        The tolerance for the failure counter.
    n_candidates_for_acquisition : int
        The number of candidates to be used for the acquisition function.

    Methods
    -------
    _compute_trust_region() -> Tuple[np.ndarray, np.ndarray]
        Computes the lower and upper bounds of the trust region.
    _optimize_acquisition_function(acquisition_function: Union[ThompsonSampling, ExpectedImprovement]) -> np.ndarray
        Optimizes the acquisition function.
    _instantiate_acquisition_function(model: SingleTaskGP) -> Union[ThompsonSampling, ExpectedImprovement]
        Instantiates the acquisition function based on the given model.
    _compute_random_embedding_matrix(input_dim: int, subspace_dim: int) -> np.ndarray
        Computes an initial random embedding matrix.
    _expand_embedding_matrix_and_observations() -> Tuple[np.ndarray, np.ndarray]
        Expands the embedding matrix and the current observations.

    References
    ----------
    [1] Papenmeier, Leonard, Luigi Nardi, and Matthias Poloczek. “Increasing the
        Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces,”
        2022. https://openreview.net/forum?id=e4Wf6112DI.
    [2] https://botorch.org/tutorials/baxus
    """

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
        initial_subspace_dimension: int = 2,
        n_new_bins_per_dimension: int = 3,
        initial_trust_region_length: float = 0.8,
        success_tolerance: int = 3,
        failure_tolerance: int = 3,
        trust_region_min_length: float = 0.5**7,
        trust_region_max_length: float = 1.6,
        n_candidates_for_acquisition: int = 100,
    ):
        """
        Initialize the BAxUS solver.

        Parameters:
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial input points for the optimization.
        y0 : np.ndarray
            The corresponding function values for the initial input points.
        mean : Mean, optional
            The mean function of the Gaussian process model. Default is None.
        kernel : Kernel, optional
            The kernel function of the Gaussian process model. Default is None.
        acq_function : Type[AcquisitionFunction], optional
            The type of acquisition function to be used. Default is ThompsonSampling.
        bounds : Tuple[float, float], optional
            The lower and upper bounds of the input space. Default is (-2.0, 2.0).
        penalize_nans_with : float, optional
            The value to penalize NaN function values with. Default is -10.
        initial_subspace_dimension : int, optional
            The initial dimension of the subspace. Default is 2.
        n_new_bins_per_dimension : int, optional
            The number of new bins to split each dimension into. Default is 3.
        initial_trust_region_length : float, optional
            The initial length of the trust region. Default is 0.8.
        success_tolerance : int, optional
            The tolerance for the success counter. Default is 3.
        failure_tolerance : int, optional
            The tolerance for the failure counter. Default is 3.
        trust_region_min_length : float, optional
            The minimum length of the trust region. Default is 0.5**7.
        trust_region_max_length : float, optional
            The maximum length of the trust region. Default is 1.6.
        n_candidates_for_acquisition : int, optional
            The number of candidates to be used for the acquisition function. Default is 100.
        """
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

        # These will hold the model and the trust region's length.
        self.gp_model = None

        # These will hold the initial and current trust region lengths.
        # The trust region length will be updated adaptively, and will
        # eventually be re-set to the initial value if it falls too low.
        self.initial_trust_region_length = initial_trust_region_length
        self.trust_region_length = initial_trust_region_length
        self.trust_region_min_length = trust_region_min_length
        self.trust_region_max_length = trust_region_max_length

        # These will hold the subspace dimension and the number of new bins per dimension.
        self.input_dimension = x0.shape[1]
        self.subspace_dimension = initial_subspace_dimension
        self.n_new_bins_per_dimension = n_new_bins_per_dimension

        # As part of the init, we should be projecting the given x0 into
        # the subspace by multiplying it by the embedding matrix.
        # In other words, we should be overwriting the history, and keeping
        # track of the observations in the subspace.
        self.embedding_matrix = self._compute_random_embedding_matrix(
            input_dim=x0.shape[1],
            subspace_dim=initial_subspace_dimension,
        )
        z0 = x0 @ self.embedding_matrix.T
        self.history["x"] = [z0]

        # These observations will be conserved with the original dimension
        # in which they were evaluated. history["x"], on the other hand,
        # will have its dimensions augmented to match the subspace
        # dimension.
        self.original_observations = [z0]

        # These will hold the success and failure counters, which
        # will be used to adaptively expand or shrink the trust region.
        self.success_counter = 0
        self.failure_counter = 0

        # We also include a tolerance for the success and failure counters.
        # TODO: Ideally, we would be including the adaptive failure
        # tolerance described in Sec. 3.4. of the BAxUS paper.
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance

        # The number of candidates to be used for the acquisition function.
        self.n_candidates_for_acquisition = n_candidates_for_acquisition

    def _compute_trust_region(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the lower and upper bounds of the trust region.

        This implementation is taken from the function "create_candidate" in [1].

        References
        ----------
        [1] https://botorch.org/tutorials/baxus
        """
        X, Y = self.get_history_as_arrays()

        # Transform these to torch
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

        # Scale the TR to be proportional to the lengthscales
        x_center = self.get_best_solution()
        x_center = torch.from_numpy(x_center).float()
        weights = self.gp_model.covar_module.base_kernel.lengthscale.detach().view(-1)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # We compute the trust regions, and clamp them to the bounds.
        tr_lb = torch.clamp(x_center - weights * self.trust_region_length, *self.bounds)
        tr_ub = torch.clamp(x_center + weights * self.trust_region_length, *self.bounds)

        return tr_lb, tr_ub

    def _optimize_acquisition_function(
        self, acquisition_function: Union[ThompsonSampling, ExpectedImprovement]
    ) -> np.ndarray:
        """Optimizes the acquisition function.

        This implementation is based on the function "create_candidate" in [1].

        Parameters
        ----------
        acquisition_function : Union[ThompsonSampling, ExpectedImprovement]
            Acquisition function to be optimized.

        Returns
        -------
        candidate : np.ndarray
            Next candidate.

        References
        ----------
        [1] https://botorch.org/tutorials/baxus
        """
        if isinstance(acquisition_function, ThompsonSampling):
            return acquisition_function(self.subspace_dimension).numpy(force=True)
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
    ) -> Union[ThompsonSampling, ExpectedImprovement]:
        """
        Instantiate the acquisition function based on the given model.

        Parameters:
        ----------
            model (SingleTaskGP): The Gaussian process model.

        Returns:
        --------
            Union[ThompsonSampling, ExpectedImprovement]: The instantiated acquisition function.

        Raises:
        -------
            NotImplementedError: If the acquisition function is neither ThompsonSampling nor ExpectedImprovement.
        """
        if self.acq_function == ThompsonSampling:
            thomspon_sampling_acq_function = self.acq_function(
                model=model,
                trust_region=self._compute_trust_region(),
                n_candidates=self.n_candidates_for_acquisition,
            )
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
        [1] Papenmeier, Leonard, Luigi Nardi, and Matthias Poloczek. “Increasing the
            Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces,”
            2022. https://openreview.net/forum?id=e4Wf6112DI.

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        References
        ----------
        [1] Papenmeier, Leonard, Luigi Nardi, and Matthias Poloczek. “Increasing the
            Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces,”
            2022. https://openreview.net/forum?id=e4Wf6112DI.
        [2] https://botorch.org/tutorials/baxus
        """
        current_observations, _ = self.get_history_as_arrays()
        embedding_matrix = self.embedding_matrix
        n_new_bins = self.n_new_bins_per_dimension

        assert (
            current_observations.shape[1] == embedding_matrix.shape[0]
        ), "Observations don't lie in row space of S"

        # If the current observations and embedding matrix are
        # already of input_dimension, then we skip.
        if self.subspace_dimension == self.input_dimension:
            return

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

        # Update the embedding matrix and the observations.
        self.embedding_matrix = embedding_matrix_update
        self.history["x"] = [arr.reshape(1, -1) for arr in observations_update]
        self.subspace_dimension = len(self.embedding_matrix)

        # Check if the current subspace dimensions is >= to
        # the input dimension. If so, then we should be replacing
        # the embedding matrix with the identity matrix.
        if self.subspace_dimension >= self.input_dimension:
            # Re-compute the current observations, and store them
            # in the history as (b, self.input_dimension) arrays.
            self.history["x"] = [
                x.reshape(1, -1) @ self.embedding_matrix for x in observations_update
            ]

            # From now on, we'll work on self.input_dimension dimensions,
            # so we no longer need to project.
            self.embedding_matrix = np.eye(self.input_dimension)

            # We also need to update the subspace dimension.
            self.subspace_dimension = self.input_dimension

    def _fit_model(
        self, model: type[SingleTaskGP], x: np.ndarray, y: np.ndarray
    ) -> SingleTaskGP:
        """Fits a single-task Gaussian Process to the data.

        This implementation is based on the optimization loop of the BAxUS
        tutorial of BoTorch [2].

        The default Gaussian Process used is a SingleTaskGP with a Matern kernel
        and a Gaussian likelihood. The Matern kernel has a fixed nu=2.5, and
        the lengthscale is constrained to be between 0.005 and 10. The outputscale
        is constrained to be between 0.05 and 10. The noise is constrained to be
        between 1e-8 and 1e-3. These constraints are, according to the authors of
        the tutorial, taken from the TuRBO paper [3].

        Parameters
        ----------
        model : type[SingleTaskGP]
            Type of the model to be fitted.
        x : np.ndarray
            Input data.
        y : np.ndarray
            Output data.

        Returns
        -------
        model : SingleTaskGP
            Fitted model.

        References
        ----------
        [1] Papenmeier, Leonard, Luigi Nardi, and Matthias Poloczek. “Increasing the
            Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces,”
            2022. https://openreview.net/forum?id=e4Wf6112DI.
        [2] https://botorch.org/tutorials/baxus
        [3] Eriksson, David, et al. “Scalable Global Optimization via Local Bayesian
            Optimization.” Advances in Neural Information Processing Systems, vol. 32,
            2019.
        """
        # Mapping to torch
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Defining the model and the likelihood
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        if self.kernel is None:
            kernel = (
                ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=self.subspace_dimension,
                        lengthscale_constraint=Interval(0.005, 10),
                    ),
                    outputscale_constraint=Interval(0.05, 10),
                )
            )
        else:
            kernel = self.kernel

        # The following code is taken verbatim from the BAxUS tutorial of BoTorch [2].
        model = SingleTaskGP(
            x, y, mean_module=self.mean, covar_module=kernel, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            # Fit the model
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                # Right after increasing the target dimensionality, the covariance matrix becomes indefinite
                # In this case, the Cholesky decomposition might fail due to numerical instabilities
                # In this case, we revert to Adam-based optimization
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

                for _ in range(100):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = -mll(output, y.flatten())
                    loss.backward()
                    optimizer.step()

        model.eval()

        return model

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Updates the BAxUS state after a new observation.

        Adapting the original implementation in the BAxUS tutorial of BoTorch [1],
        we update the trust region after each observation, and we also update the
        embedding matrix and the observations if the trust region falls below a
        certain threshold (as per the usual TuRBO update rules, see [2]).

        Parameters
        ----------
        x : np.ndarray
            input point.
        y : np.ndarray
            output objective value for x.

        References
        ----------
        [1] https://botorch.org/tutorials/baxus
        [2] Eriksson, David, et al. “Scalable Global Optimization via Local Bayesian
            Optimization.” Advances in Neural Information Processing Systems, vol. 32,
            2019.
        """
        self.original_observations.append(x)

        # This code is mostly taken verbatim from the BAxUS tutorial of BoTorch [1].
        # It has been adapted to work with the current implementation of the
        # BAxUS algorithm.
        previous_best_value = self.get_best_performance(until=-1)
        if max(y) > previous_best_value + 1e-3 * (previous_best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.trust_region_length = min(
                2.0 * self.trust_region_length, self.trust_region_max_length
            )
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.trust_region_length /= 2.0
            self.failure_counter = 0

        if self.trust_region_length < self.trust_region_min_length:
            self._expand_embedding_matrix_and_observations()
            self._reset_trust_region()

    def _reset_trust_region(self) -> None:
        """Resets the trust region to its initial value.

        This method is called when the trust region falls below a certain threshold,
        and it resets the trust region to its initial value. It also updates the
        success and failure counters.
        """
        self.trust_region_length = self.initial_trust_region_length
        self.failure_counter = 0
        self.success_counter = 0

    def next_candidate(self) -> np.ndarray:
        """Computes the next candidate.

        This implementation is based on the function "create_candidate" in [1].

        Returns
        -------
        candidate : np.ndarray
            Next candidate.

        References
        ----------
        [1] https://botorch.org/tutorials/baxus
        """
        # Build up the history
        X, Y = self.get_history_as_arrays()

        # Penalize the NaNs in the objective function by assigning them a value
        # stored in self.penalize_nans_with.
        Y[np.isnan(Y)] = self.penalize_nans_with

        # Fit a GP
        model = self._fit_model(SingleTaskGP, X, Y)

        # Update the stored model
        self.gp_model = model

        # Instantiate the acquisition function
        acq_function = self._instantiate_acquisition_function(model=model)

        # Optimize the acquisition function
        candidate = self._optimize_acquisition_function(
            acquisition_function=acq_function
        )

        # The update state is handled by post_update, which will run automatically
        # as part of the abstract step method.
        return candidate
