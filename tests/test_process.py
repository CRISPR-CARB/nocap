"""Test functions for the process module."""

import pytest
import torch

from nocap.process import (
    apply_bernoulli_lognormal_outlier_process,
    apply_lognormal_noise_process,
    apply_poisson_process,
    apply_quantile_logistic_dropout_process,
    apply_row_normalization_and_lognormal_scaling_process,
    generate_test_data_tensor,
)


def test_generate_test_data_tensor_shape():
    """Test tensor shape."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    if tensor.shape != (100, 200):
        pytest.fail("Tensor shape mismatch")


def test_generate_test_data_tensor_seed():
    """Test tensor generation with random seed."""
    tensor1 = generate_test_data_tensor(seed=42)
    tensor2 = generate_test_data_tensor(seed=42)
    tensor3 = generate_test_data_tensor(seed=43)
    if not torch.equal(tensor1, tensor2):
        pytest.fail("Tensors generated with the same seed are not equal")
    if torch.equal(tensor1, tensor3):
        pytest.fail("Tensors generated with different seeds are equal")


def test_generate_test_data_tensor_custom_params():
    """Test tensor generation with custom parameters."""
    tensor = generate_test_data_tensor(
        n_rows=50, n_cols=100, scale_dist_params=(1, 0.5), loc_dist_params=(3, 25)
    )
    if tensor.shape != (50, 100):
        pytest.fail("Tensor shape mismatch with custom parameters")
    if tensor.dtype != torch.float:
        pytest.fail("Tensor dtype mismatch with custom parameters")


def test_generate_test_data_tensor_values():
    """Test tensor values for NaN, Inf, and negativity."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    if torch.isnan(tensor).any():
        pytest.fail("Tensor contains NaN values")
    if torch.isinf(tensor).any():
        pytest.fail("Tensor contains Inf or -Inf values")
    if not (tensor >= 0).all():
        pytest.fail("Tensor contains negative values")


def test_generate_test_data_tensor_column_row_uniqueness():
    """Test uniqueness of tensor columns and rows."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    for col in tensor.T:
        if torch.unique(col).numel() <= 1:
            pytest.fail("Column contains identical values")
    for row in tensor:
        if torch.unique(row).numel() <= 1:
            pytest.fail("Row contains identical values")


def test_generate_test_data_tensor_column_stats():
    """Test statistics of tensor columns."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    means = tensor.mean(dim=0)
    stds = tensor.std(dim=0)
    if len(torch.unique(means)) <= 1:
        pytest.fail("Means of columns are not unique")
    if len(torch.unique(stds)) <= 1:
        pytest.fail("Standard deviations of columns are not unique")


def test_apply_lognormal_noise_process_shape():
    """Test shape of tensor after applying lognormal noise."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    noisy_tensor = apply_lognormal_noise_process(tensor)
    if noisy_tensor.shape != tensor.shape:
        pytest.fail("Shape mismatch after applying lognormal noise")


def test_apply_lognormal_noise_process_different_noise():
    """Test that noise applied is different for different seeds."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    noisy_tensor1 = apply_lognormal_noise_process(tensor, mean=0.0, std=1.0)
    noisy_tensor2 = apply_lognormal_noise_process(tensor, mean=0.0, std=2.0)
    if torch.equal(noisy_tensor1, noisy_tensor2):
        pytest.fail("Noisy tensors are equal for different noise applications")


def test_apply_lognormal_noise_process_mean_std():
    """Test mean and std of noise applied to tensor."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    noisy_tensor = apply_lognormal_noise_process(tensor, mean=0.0, std=1.0)
    noise = noisy_tensor - tensor
    log_noise = torch.log(noise)
    if not torch.isclose(log_noise.mean(), torch.tensor(0.0), atol=0.1):
        pytest.fail("Mean of noise is not close to expected mean")
    if not torch.isclose(log_noise.std(), torch.tensor(1.0), atol=0.1):
        pytest.fail("Standard deviation of noise is not close to expected std")


def test_apply_lognormal_noise_process_data_values():
    """Test that no NaN, Inf, or negative values are introduced by noise."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    noisy_tensor = apply_lognormal_noise_process(tensor)
    if torch.isnan(noisy_tensor).any():
        pytest.fail("Noisy tensor contains NaN values")
    if torch.isinf(noisy_tensor).any():
        pytest.fail("Noisy tensor contains Inf or -Inf values")
    if not (noisy_tensor >= 0).all():
        pytest.fail("Noisy tensor contains negative values")


def test_apply_bernoulli_lognormal_outlier_process_shape():
    """Test shape after applying outlier process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=0.0, sigma=1.0)
    if outlier_tensor.shape != tensor.shape:
        pytest.fail("Shape mismatch after applying outlier process")


def test_apply_bernoulli_lognormal_outlier_process_no_outliers():
    """Test no outliers when pi is 0."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.0, mu=0.0, sigma=1.0)
    if not torch.equal(tensor, outlier_tensor):
        pytest.fail("Tensor changed when pi is ~0")


def test_apply_bernoulli_lognormal_outlier_process_all_outliers():
    """Test all outliers when pi is 1."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=1.0, mu=0.0, sigma=1.0)
    if torch.equal(tensor, outlier_tensor):
        pytest.fail("Tensor did not change when pi is ~1")


def test_apply_bernoulli_lognormal_outlier_process_values():
    """Test values for NaN, Inf, and negativity."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=0.0, sigma=1.0)
    if torch.isnan(outlier_tensor).any():
        pytest.fail("Outlier tensor contains NaN values")
    if torch.isinf(outlier_tensor).any():
        pytest.fail("Outlier tensor contains Inf or -Inf values")
    if not (outlier_tensor >= 0).all():
        pytest.fail("Outlier tensor contains negative values")


def test_apply_bernoulli_lognormal_outlier_process_different_mu_sigma():
    """Test different mu and sigma values produce different outliers."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    outlier_tensor1 = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=0.0, sigma=1.0)
    outlier_tensor2 = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=1.0, sigma=2.0)
    if torch.equal(outlier_tensor1, outlier_tensor2):
        pytest.fail("Outlier tensors are equal for different mu and sigma values")


def test_apply_bernoulli_lognormal_outlier_process_outlier_range():
    """Test outliers are within an expected range."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    pi, mu, sigma = 0.1, 0.0, 1.0
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=pi, mu=mu, sigma=sigma)
    log_outlier_tensor = torch.log(outlier_tensor)

    # Check that 95% of the data falls within the 2*sigma range
    mean = torch.mean(log_outlier_tensor)
    std_dev = torch.std(log_outlier_tensor)
    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev
    within_range = (log_outlier_tensor >= lower_bound) & (log_outlier_tensor <= upper_bound)
    proportion_within_range = torch.sum(within_range).item() / log_outlier_tensor.numel()
    if proportion_within_range < 0.95:
        pytest.fail(
            f"{proportion_within_range:.2%} (should be >95%) of outlier values are within 2 sigma of the mean."
        )


def test_apply_row_normalization_and_lognormal_scaling_shape():
    """Test shape after normalization and scaling."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    normalized_scaled_tensor = apply_row_normalization_and_lognormal_scaling_process(
        tensor, mu=0.0, sigma=1.0
    )
    if normalized_scaled_tensor.shape != tensor.shape:
        pytest.fail("Shape mismatch after normalization and scaling")


def test_apply_row_normalization_and_lognormal_scaling_no_nan_inf_negative():
    """Test no NaN, Inf, or negative values after normalization and scaling."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    normalized_scaled_tensor = apply_row_normalization_and_lognormal_scaling_process(
        tensor, mu=0.0, sigma=1.0
    )
    if torch.isnan(normalized_scaled_tensor).any():
        pytest.fail("Normalized and scaled tensor contains NaN values")
    if torch.isinf(normalized_scaled_tensor).any():
        pytest.fail("Normalized and scaled tensor contains Inf or -Inf values")
    if not (normalized_scaled_tensor >= 0).all():
        pytest.fail("Normalized and scaled tensor contains negative values")


def test_apply_row_normalization_and_lognormal_scaling_values():
    """Test scaling factors after normalization and scaling."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=300)
    mu, sigma = 0.0, 1.0
    epsilon = 1e-6  # Small value to avoid numerical issues
    normalized_scaled_tensor = apply_row_normalization_and_lognormal_scaling_process(
        tensor, mu=mu, sigma=sigma
    )

    # Get expected and actual (inferred) scaling factors
    row_sums = tensor.sum(dim=1, keepdim=True)
    normalized_tensor = tensor / row_sums
    inferred_scaling_factors = normalized_scaled_tensor / normalized_tensor
    log_inferred_scaling_factors = torch.log(inferred_scaling_factors)
    expected_scaling_factors = torch.distributions.LogNormal(mu, sigma).sample([tensor.size(0), 1])
    log_expected_scaling_factors = torch.log(expected_scaling_factors + epsilon)

    # Calculate the expected 2*sigma data range where ~95% of the data should fall
    mean_expected = log_expected_scaling_factors.mean().item()
    std_expected = log_expected_scaling_factors.std().item()
    lower_bound = mean_expected - 2 * std_expected
    upper_bound = mean_expected + 2 * std_expected
    within_2sigma = (
        (
            (log_inferred_scaling_factors >= lower_bound)
            & (log_inferred_scaling_factors <= upper_bound)
        )
        .float()
        .mean()
        .item()
    )
    if within_2sigma < 0.95:
        pytest.fail(
            f"{within_2sigma:.2%} (should be >95%) of outlier values are within 2 sigma of the mean."
        )


def test_apply_row_normalization_and_lognormal_scaling_different_mu_sigma():
    """Test different mu and sigma values produce different normalized and scaled tensors."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    normalized_scaled_tensor1 = apply_row_normalization_and_lognormal_scaling_process(
        tensor, mu=0.0, sigma=1.0
    )
    normalized_scaled_tensor2 = apply_row_normalization_and_lognormal_scaling_process(
        tensor, mu=1.0, sigma=2.0
    )
    if torch.equal(normalized_scaled_tensor1, normalized_scaled_tensor2):
        pytest.fail("Normalized and scaled tensors are equal for different mu and sigma values")


def test_apply_quantile_logistic_dropout_process_shape():
    """Test shape after applying dropout process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    if dropout_tensor.shape != tensor.shape:
        pytest.fail("Shape mismatch after applying dropout process")


def test_apply_quantile_logistic_dropout_process_no_nan_inf_negative():
    """Test no NaN, Inf, or negative values after dropout process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    if torch.isnan(dropout_tensor).any():
        pytest.fail("Dropout tensor contains NaN values")
    if torch.isinf(dropout_tensor).any():
        pytest.fail("Dropout tensor contains Inf or -Inf values")
    if not (dropout_tensor >= 0).all():
        pytest.fail("Dropout tensor contains negative values")


def test_apply_quantile_logistic_dropout_process_different_params():
    """Test different parameters produce different dropout tensors."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    dropout_tensor1 = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    dropout_tensor2 = apply_quantile_logistic_dropout_process(tensor, k=2.0, q=0.5)
    if torch.equal(dropout_tensor1, dropout_tensor2):
        pytest.fail("Dropout tensors are equal for different k values")
    dropout_tensor3 = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.7)
    if torch.equal(dropout_tensor1, dropout_tensor3):
        pytest.fail("Dropout tensors are equal for different q values")


def test_apply_quantile_logistic_dropout_process_expected_range():
    """Test dropout values are within expected range."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    if dropout_tensor.min().item() < 0:
        pytest.fail("Dropout tensor contains values below 0")
    if dropout_tensor.max().item() > tensor.max().item():
        pytest.fail("Dropout tensor contains values above original tensor max")


def test_apply_quantile_logistic_dropout_process_all_dropout():
    """Test that all values dropout when pi is 1."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=100, q=0.99)
    drop_out_ratio = (dropout_tensor == 0).float().mean().item()
    if drop_out_ratio <= 0.95:
        pytest.fail(f"Expected >95% dropout, but got {drop_out_ratio * 100}%")


def test_apply_quantile_logistic_dropout_process_no_dropout():
    """Test that no values dropout when pi is 0."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=100, q=0.01)
    drop_out_ratio = (dropout_tensor == 0).float().mean().item()
    if drop_out_ratio >= 0.05:
        pytest.fail(f"Expected <5% dropout, but got {drop_out_ratio * 100}%")


def test_apply_quantile_logistic_dropout_process_more_dropouts():
    """Test that there are more dropouts after applying the process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    original_nonzero_count = (tensor != 0).sum().item()
    dropout_nonzero_count = (dropout_tensor != 0).sum().item()
    if dropout_nonzero_count >= original_nonzero_count:
        pytest.fail("There are not more dropouts after applying the process")


def test_apply_poisson_process_shape():
    """Test shape after applying Poisson process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    poisson_tensor = apply_poisson_process(tensor)
    if poisson_tensor.shape != tensor.shape:
        pytest.fail("Shape mismatch after applying Poisson process")


def test_apply_poisson_process_no_nan_inf_negative():
    """Test no NaN, Inf, or negative values after Poisson process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    poisson_tensor = apply_poisson_process(tensor)
    if torch.isnan(poisson_tensor).any():
        pytest.fail("Poisson tensor contains NaN values")
    if torch.isinf(poisson_tensor).any():
        pytest.fail("Poisson tensor contains Inf or -Inf values")
    if not (poisson_tensor >= 0).all():
        pytest.fail("Poisson tensor contains negative values")


def test_apply_poisson_process_different_inputs():
    """Test different inputs give different outputs."""
    tensor1 = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    tensor2 = generate_test_data_tensor(n_rows=100, n_cols=200, seed=43)
    poisson_tensor1 = apply_poisson_process(tensor1)
    poisson_tensor2 = apply_poisson_process(tensor2)
    if torch.equal(poisson_tensor1, poisson_tensor2):
        pytest.fail("Poisson tensors are equal for different inputs")


def test_apply_poisson_process_int_values():
    """Test that Poisson sampling produces integer values."""
    tensor = generate_test_data_tensor(n_rows=200, n_cols=200)
    poisson_tensor = apply_poisson_process(tensor)
    if not torch.all(poisson_tensor == poisson_tensor.int()):
        pytest.fail("Some sampled values are not integers")
