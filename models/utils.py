import torch

def create_lagged_targets(targets, lag_steps):
    """
    Creates a tensor of lagged target values based on specified lag steps.

    Args:
        targets (torch.Tensor): Time series of target values of shape [num_timesteps].
        lag_steps (list[int]): List of lag steps to use (e.g., [3, 6, 12]).

    Returns:
        torch.Tensor: Lagged target tensor of shape [num_timesteps - max(lag_steps), len(lag_steps)].
    """
    max_lag = max(lag_steps)
    num_timesteps = targets.size(0)

    if num_timesteps <= max_lag:
        raise ValueError("Not enough timesteps in the targets to construct lagged features.")

    lagged_data = [
        targets[max_lag - lag:num_timesteps - lag].unsqueeze(-1) for lag in lag_steps
    ]

    lagged_targets = torch.cat(lagged_data, dim=-1)
    return lagged_targets
