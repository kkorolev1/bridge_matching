import torch


def get_timesteps_uniform(params, device):
    num_steps = params["num_steps"]
    return torch.linspace(1e-4, 1, num_steps + 1, device=device)


def get_timesteps_diff(params, device):
    num_steps = params["num_steps"]
    sigma_min, sigma_max = params["sigma_min"], params["sigma_max"]
    rho = params["rho"]
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    t_steps = torch.flip(t_steps, (0,))
    return t_steps
