"""Test functions for the process module."""

import numpy as np
import pandas as pd
import pytest
import torch

from nocap.process import (
    apply_bernoulli_lognormal_outlier_process,
    apply_lognormal_noise_process,
    apply_poisson_process,
    apply_quantile_logistic_dropout_process,
    apply_row_normalization_and_lognormal_scaling,
    generate_test_data_tensor,
)


def test_generate_test_data_tensor_shape():
    """Test tensor shape."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    assert tensor.shape == (100, 200), "Tensor shape mismatch"


def test_generate_test_data_tensor_seed():
    """Test tensor generation with random seed."""
    tensor1 = generate_test_data_tensor(seed=42)
    tensor2 = generate_test_data_tensor(seed=42)
    tensor3 = generate_test_data_tensor(seed=43)
    assert torch.equal(tensor1, tensor2), "Tensors generated with the same seed are not equal"
    assert not torch.equal(tensor1, tensor3), "Tensors generated with different seeds are equal"


def test_generate_test_data_tensor_custom_params():
    """Test tensor generation with custom parameters."""
    tensor = generate_test_data_tensor(
        n_rows=50, n_cols=100, scale_dist_params=(1, 0.5), loc_dist_params=(3, 25)
    )
    assert tensor.shape == (50, 100), "Tensor shape mismatch with custom parameters"
    assert tensor.dtype == torch.float, "Tensor dtype mismatch with custom parameters"


def test_generate_test_data_tensor_values():
    """Test tensor values for NaN, Inf, and negativity."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf or -Inf values"
    assert (tensor >= 0).all(), "Tensor contains negative values"


def test_generate_test_data_tensor_column_row_uniqueness():
    """Test uniqueness of tensor columns."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    for col in tensor.T:
        assert len(torch.unique(col)) > 1, "Column contains identical values"
    for row in tensor:
        assert len(torch.unique(row)) > 1, "Row contains identical values"


def test_generate_test_data_tensor_column_stats():
    """Test statistics of tensor columns."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    means = tensor.mean(dim=0)
    stds = tensor.std(dim=0)
    assert len(torch.unique(means)) > 1, "Means of columns are not unique"
    assert len(torch.unique(stds)) > 1, "Standard deviations of columns are not unique"


def test_apply_lognormal_noise_process_shape():
    """Test shape of tensor after applying lognormal noise."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    noisy_tensor = apply_lognormal_noise_process(tensor)
    assert noisy_tensor.shape == tensor.shape, "Shape mismatch after applying lognormal noise"


def test_apply_lognormal_noise_process_different_noise():
    """Test that noise applied is different for different seeds."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    noisy_tensor1 = apply_lognormal_noise_process(tensor, mean=0.0, std=1.0)
    noisy_tensor2 = apply_lognormal_noise_process(tensor, mean=0.0, std=2.0)
    assert not torch.equal(
        noisy_tensor1, noisy_tensor2
    ), "Noisy tensors are equal for different noise applications"


def test_apply_lognormal_noise_process_mean_std():
    """Test mean and std of noise applied to tensor."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    noisy_tensor = apply_lognormal_noise_process(tensor, mean=0.0, std=1.0)
    noise = noisy_tensor - tensor
    log_noise = torch.log(noise)
    assert torch.isclose(
        log_noise.mean(), torch.tensor(0.0), atol=0.1
    ), "Mean of noise is not close to expected mean"
    assert torch.isclose(
        log_noise.std(), torch.tensor(1.0), atol=0.1
    ), "Standard deviation of noise is not close to expected std"


def test_apply_lognormal_noise_process_data_values():
    """Test that no NaN, Inf, or negative values are introduced by noise."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    noisy_tensor = apply_lognormal_noise_process(tensor)
    assert not torch.isnan(noisy_tensor).any(), "Noisy tensor contains NaN values"
    assert not torch.isinf(noisy_tensor).any(), "Noisy tensor contains Inf or -Inf values"
    assert (noisy_tensor >= 0).all(), "Noisy tensor contains negative values"


def test_apply_bernoulli_lognormal_outlier_process_shape():
    """Test shape after applying outlier process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=0.0, sigma=1.0)
    assert outlier_tensor.shape == tensor.shape, "Shape mismatch after applying outlier process"


def test_apply_bernoulli_lognormal_outlier_process_no_outliers():
    """Test no outliers when pi is 0."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.0, mu=0.0, sigma=1.0)
    assert torch.equal(tensor, outlier_tensor), "Tensor changed when pi is 0"


def test_apply_bernoulli_lognormal_outlier_process_all_outliers():
    """Test all outliers when pi is 1."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=1.0, mu=0.0, sigma=1.0)
    assert not torch.equal(tensor, outlier_tensor), "Tensor did not change when pi is 1"


def test_apply_bernoulli_lognormal_outlier_process_values():
    """Test values for NaN, Inf, and negativity."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=0.0, sigma=1.0)
    assert not torch.isnan(outlier_tensor).any(), "Outlier tensor contains NaN values"
    assert not torch.isinf(outlier_tensor).any(), "Outlier tensor contains Inf or -Inf values"
    assert (outlier_tensor >= 0).all(), "Outlier tensor contains negative values"


def test_apply_bernoulli_lognormal_outlier_process_different_mu_sigma():
    """Test different mu and sigma values produce different outliers."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    outlier_tensor1 = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=0.0, sigma=1.0)
    outlier_tensor2 = apply_bernoulli_lognormal_outlier_process(tensor, pi=0.1, mu=1.0, sigma=2.0)
    assert not torch.equal(
        outlier_tensor1, outlier_tensor2
    ), "Outlier tensors are equal for different mu and sigma values"


def test_apply_bernoulli_lognormal_outlier_process_outlier_range():
    """Test outliers are within an expected range."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    pi, mu, sigma = 0.1, 0.0, 1.0
    outlier_tensor = apply_bernoulli_lognormal_outlier_process(tensor, pi=pi, mu=mu, sigma=sigma)

    # Calculate the expected range for outliers (mu +/- 3*sigma for Normal distribution)
    expected_min = (tensor * torch.exp(torch.tensor(mu - 3 * sigma))).min().item()
    expected_max = (tensor * torch.exp(torch.tensor(mu + 3 * sigma))).max().item()

    assert (
        outlier_tensor.min().item() >= expected_min
    ), f"Outlier tensor contains values below expected range: {outlier_tensor.min().item()} < {expected_min}"
    assert (
        outlier_tensor.max().item() <= expected_max
    ), f"Outlier tensor contains values above expected range: {outlier_tensor.max().item()} > {expected_max}"


def test_apply_row_normalization_and_lognormal_scaling_shape():
    """Test shape after normalization and scaling."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    normalized_scaled_tensor = apply_row_normalization_and_lognormal_scaling(
        tensor, mu=0.0, sigma=1.0
    )
    assert (
        normalized_scaled_tensor.shape == tensor.shape
    ), "Shape mismatch after normalization and scaling"


def test_apply_row_normalization_and_lognormal_scaling_no_nan_inf_negative():
    """Test no NaN, Inf, or negative values after normalization and scaling."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    normalized_scaled_tensor = apply_row_normalization_and_lognormal_scaling(
        tensor, mu=0.0, sigma=1.0
    )
    assert not torch.isnan(
        normalized_scaled_tensor
    ).any(), "Normalized and scaled tensor contains NaN values"
    assert not torch.isinf(
        normalized_scaled_tensor
    ).any(), "Normalized and scaled tensor contains Inf or -Inf values"
    assert (
        normalized_scaled_tensor >= 0
    ).all(), "Normalized and scaled tensor contains negative values"


### TODO: fix this test
# def test_apply_row_normalization_and_lognormal_scaling_values():
#     """Test scaling factors and data range after normalization and scaling."""
#     tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
#     mu, sigma = 0.0, 1.0
#     normalized_scaled_tensor = apply_row_normalization_and_lognormal_scaling(tensor, mu=mu, sigma=sigma)

#     # Normalize the original tensor row-wise in the test
#     row_sums = tensor.sum(dim=1, keepdim=True)
#     normalized_tensor = tensor / row_sums

#     # Calculate the scaling factors applied by comparing normalized tensor and the output
#     inferred_scaling_factors = normalized_scaled_tensor / normalized_tensor

#     # Statistical properties verification: mean and standard deviation in log space
#     log_inferred_scaling_factors = torch.log(inferred_scaling_factors)

#     mean_inferred = log_inferred_scaling_factors.mean().item()
#     std_inferred = log_inferred_scaling_factors.std().item()

#     # Allow some tolerance due to statistical fluctuations and sample size
#     tolerance = 0.1
#     assert abs(mean_inferred - mu) < tolerance, f"Mean of inferred scaling factors ({mean_inferred}) not close to expected mean ({mu})"
#     assert abs(std_inferred - sigma) < tolerance, f"Std of inferred scaling factors ({std_inferred}) not close to expected std ({sigma})"

#     # Calculate the expected range values based on the statistical properties using 3 sigma rule
#     expected_min = torch.exp(torch.tensor(mu - 3 * sigma)).item()
#     expected_max = torch.exp(torch.tensor(mu + 3 * sigma)).item()

#     # Ensure the inferred scaling factors are within the expected range
#     assert inferred_scaling_factors.min().item() >= expected_min, f"Normalized and scaled tensor contains values below expected range: {inferred_scaling_factors.min().item()} < {expected_min}"
#     assert inferred_scaling_factors.max().item() <= expected_max, f"Normalized and scaled tensor contains values above expected range: {inferred_scaling_factors.max().item()} > {expected_max}"

#     # Verifying the range of the data values in the normalized and scaled tensor
#     data_min_expected = (tensor.min() / row_sums.max()).item() * expected_min
#     data_max_expected = (tensor.max() / row_sums.min()).item() * expected_max

#     assert normalized_scaled_tensor.min().item() >= data_min_expected, f"Normalized and scaled tensor contains values below expected range: {normalized_scaled_tensor.min().item()} < {data_min_expected}"
#     assert normalized_scaled_tensor.max().item() <= data_max_expected, f"Normalized and scaled tensor contains values above expected range: {normalized_scaled_tensor.max().item()} > {data_max_expected}"


def test_apply_row_normalization_and_lognormal_scaling_different_mu_sigma():
    """Test different mu and sigma values produce different normalized and scaled tensors."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    normalized_scaled_tensor1 = apply_row_normalization_and_lognormal_scaling(
        tensor, mu=0.0, sigma=1.0
    )
    normalized_scaled_tensor2 = apply_row_normalization_and_lognormal_scaling(
        tensor, mu=1.0, sigma=2.0
    )
    assert not torch.equal(
        normalized_scaled_tensor1, normalized_scaled_tensor2
    ), "Normalized and scaled tensors are equal for different mu and sigma values"


def test_apply_quantile_logistic_dropout_process_shape():
    """Test shape after applying dropout process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    assert dropout_tensor.shape == tensor.shape, "Shape mismatch after applying dropout process"


def test_apply_quantile_logistic_dropout_process_no_nan_inf_negative():
    """Test no NaN, Inf, or negative values after dropout process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    assert not torch.isnan(dropout_tensor).any(), "Dropout tensor contains NaN values"
    assert not torch.isinf(dropout_tensor).any(), "Dropout tensor contains Inf or -Inf values"
    assert (dropout_tensor >= 0).all(), "Dropout tensor contains negative values"


def test_apply_quantile_logistic_dropout_process_different_params():
    """Test different parameters produce different dropout tensors."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    dropout_tensor1 = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    dropout_tensor2 = apply_quantile_logistic_dropout_process(tensor, k=2.0, q=0.5)
    assert not torch.equal(
        dropout_tensor1, dropout_tensor2
    ), "Dropout tensors are equal for different k values"
    dropout_tensor3 = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.7)
    assert not torch.equal(
        dropout_tensor1, dropout_tensor3
    ), "Dropout tensors are equal for different q values"


def test_apply_quantile_logistic_dropout_process_expected_range():
    """Test dropout values are within expected range."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    assert dropout_tensor.min().item() >= 0, "Dropout tensor contains values below 0"
    assert (
        dropout_tensor.max().item() <= tensor.max().item()
    ), "Dropout tensor contains values above original tensor max"


def test_apply_quantile_logistic_dropout_process_all_dropout():
    """Test that all values dropout when pi is 1."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=100, q=0.99)
    drop_out_ratio = (dropout_tensor == 0).float().mean().item()
    assert drop_out_ratio > 0.95, f"Expected >95% dropout, but got {drop_out_ratio * 100}%"


def test_apply_quantile_logistic_dropout_process_no_dropout():
    """Test that no values dropout when pi is 0."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=100, q=0.01)
    drop_out_ratio = (dropout_tensor == 0).float().mean().item()
    assert drop_out_ratio < 0.05, f"Expected <5% dropout, but got {drop_out_ratio * 100}%"


def test_apply_quantile_logistic_dropout_process_more_dropouts():
    """Test that there are more dropouts after applying the process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    dropout_tensor = apply_quantile_logistic_dropout_process(tensor, k=1.0, q=0.5)
    original_nonzero_count = (tensor != 0).sum().item()
    dropout_nonzero_count = (dropout_tensor != 0).sum().item()
    assert (
        dropout_nonzero_count < original_nonzero_count
    ), "There are not more dropouts after applying the process"


def test_apply_poisson_process_shape():
    """Test shape after applying Poisson process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    poisson_tensor = apply_poisson_process(tensor)
    assert poisson_tensor.shape == tensor.shape, "Shape mismatch after applying Poisson process"


def test_apply_poisson_process_no_nan_inf_negative():
    """Test no NaN, Inf, or negative values after Poisson process."""
    tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
    poisson_tensor = apply_poisson_process(tensor)
    assert not torch.isnan(poisson_tensor).any(), "Poisson tensor contains NaN values"
    assert not torch.isinf(poisson_tensor).any(), "Poisson tensor contains Inf or -Inf values"
    assert (poisson_tensor >= 0).all(), "Poisson tensor contains negative values"


def test_apply_poisson_process_different_inputs():
    """Test different inputs give different outputs."""
    tensor1 = generate_test_data_tensor(n_rows=100, n_cols=200, seed=42)
    tensor2 = generate_test_data_tensor(n_rows=100, n_cols=200, seed=43)
    poisson_tensor1 = apply_poisson_process(tensor1)
    poisson_tensor2 = apply_poisson_process(tensor2)
    assert not torch.equal(
        poisson_tensor1, poisson_tensor2
    ), "Poisson tensors are equal for different inputs"


### TODO: fix this test
# def test_apply_poisson_process_values():
#     """Test that Poisson sampling produces values with expected statistical properties."""
#     tensor = generate_test_data_tensor(n_rows=100, n_cols=200)
#     sampled_tensor = apply_poisson_process(tensor)

#     mean_tensor = tensor.mean().item()
#     mean_sampled = sampled_tensor.float().mean().item()
#     var_sampled = sampled_tensor.float().var().item()

#     tolerance = 0.1 * mean_tensor

#     assert abs(mean_sampled - mean_tensor) < tolerance, f"Sampled mean {mean_sampled} not close to expected mean {mean_tensor}"
#     assert abs(var_sampled - mean_tensor) < tolerance, f"Sampled variance {var_sampled} not close to expected mean {mean_tensor}"
#     assert torch.all(sampled_tensor == sampled_tensor.int()), "Some sampled values are not integers"
