"""Process for data generation."""

import torch


def generate_test_data_tensor(
    n_rows=100, n_cols=200, seed=42, scale_dist_params=(0, 1), loc_dist_params=(2, 50)
):
    """
    Generate a tensor of test data with specified dimensions and random values.

    Args:
        n_rows (int, optional): Number of rows in the generated tensor. Default is 100.
        n_cols (int, optional): Number of columns in the generated tensor. Default is 200.
        seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 42.
        scale_dist_params (tuple, optional): Parameters (mean, std) for the log-normal distribution from which
            to draw the scales of each column. Default is (0, 1).
        loc_dist_params (tuple, optional): Parameters (shape, scale) for the gamma distribution from which
            to draw the locations of each column. Default is (2, 50).

    Returns:
        torch.Tensor: A tensor of shape (n_rows, n_cols) with random float values.
    """
    torch.manual_seed(seed)

    scale_mean, scale_std = scale_dist_params
    loc_shape, loc_scale = loc_dist_params

    # Generate random scales and locations for each column using PyTorch distributions
    scales = torch.distributions.LogNormal(scale_mean, scale_std).sample((n_cols,))
    locs = torch.distributions.Gamma(loc_shape, loc_scale).sample((n_cols,))

    data_array = torch.stack(
        [scale * torch.rand(n_rows) + loc for scale, loc in zip(scales, locs)], dim=1
    )

    return data_array


def apply_lognormal_noise_process(tensor, mean=0.0, std=1.0):
    """
    Apply a log-normal noise process to the input tensor.

    This function generates log-normal noise with the specified mean and standard deviation,
    and adds it to the input tensor.

    Args:
        tensor (torch.Tensor): The input tensor to which noise will be added.
        mean (float, optional): The mean of the log-normal distribution. Default is 0.0.
        std (float, optional): The standard deviation of the log-normal distribution. Default is 1.0.

    Returns:
        torch.Tensor: The tensor with added log-normal noise.
    """
    noise = torch.distributions.LogNormal(mean, std).sample(tensor.size())
    return tensor + noise


def apply_bernoulli_lognormal_outlier_process(tensor, pi, mu, sigma):
    r"""
    Apply a Bernoulli-LogNormal outlier process to the input tensor.

    This function generates binary indicators using a Bernoulli distribution with probability $\pi$,
    and scale factors using a LogNormal distribution with mean $\mu$ and standard deviation $\sigma$.
    The input tensor is then updated by multiplying it with the scale factors where the binary indicator is 1.

    Args:
        tensor (torch.Tensor): The input tensor to which the outlier process will be applied.
        pi (float): The probability of an element being an outlier (Bernoulli distribution parameter).
        mu (float): The mean of the LogNormal distribution for generating scale factors.
        sigma (float): The standard deviation of the LogNormal distribution for generating scale factors.

    Returns:
        torch.Tensor: The tensor with applied Bernoulli-LogNormal outlier process.
    """
    binary_indicators = torch.distributions.Bernoulli(pi).sample(tensor.size())
    scale_factors = torch.distributions.LogNormal(mu, sigma).sample(tensor.size())
    updated_tensor = tensor * scale_factors * binary_indicators + (1 - binary_indicators) * tensor
    return updated_tensor


def apply_row_normalization_and_lognormal_scaling_process(tensor, mu, sigma):
    r"""
    Apply row normalization and log-normal scaling to the input tensor.

    This function normalizes each row of the input tensor by dividing by the sum of the elements in that row.
    It then scales the normalized tensor by sampling from a log-normal distribution with the given mean ($\mu$)
    and standard deviation ($\sigma$).

    Args:
        tensor (torch.Tensor): The input tensor to be normalized and scaled.
        mu (float): The mean of the log-normal distribution.
        sigma (float): The standard deviation of the log-normal distribution.

    Returns:
        torch.Tensor: The normalized and scaled tensor.
    """
    row_sums = tensor.sum(dim=1, keepdim=True)
    scaling_factors = torch.distributions.LogNormal(mu, sigma).sample([tensor.size(0), 1])
    return (scaling_factors * tensor) / row_sums


def apply_quantile_logistic_dropout_process(tensor, k, q):
    """
    Apply a quantile-based logistic dropout process to the input tensor.

    This function calculates dropout probabilities using a logistic function centered at the quantile threshold
    of the logarithmically transformed tensor values. It then applies dropout based on these probabilities.

    Args:
        tensor (torch.Tensor): The input tensor to which the dropout process will be applied.
        k (float): The slope parameter controlling the steepness of the logistic function.
        q (float): The quantile threshold for determining the central point of the logistic function.

    Returns:
        torch.Tensor: The tensor after applying the dropout process.
    """
    log_y = torch.log(tensor + 1)  # log transformation of the data
    log_y_0 = torch.quantile(log_y, q)
    pi = 1 / (1 + torch.exp(-k * (log_y - log_y_0)))
    binary_indicators = torch.distributions.Bernoulli(pi).sample()
    return binary_indicators * tensor


def apply_poisson_process(tensor):
    r"""
    Apply a Poisson process to the input tensor.

    This function takes a tensor as input and generates samples from a Poisson distribution
    with the rate parameter ($\lambda$) given by the values in the input tensor. The resulting
    tensor contains the sampled values.

    Args:
        tensor (torch.Tensor): A tensor containing the rate parameters ($\lambda$) for the Poisson distribution.

    Returns:
        torch.Tensor: A tensor containing the sampled values from the Poisson distribution.
    """
    return torch.distributions.Poisson(tensor).sample()
