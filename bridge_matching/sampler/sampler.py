import torch

from .timesteps import get_timesteps_uniform, get_timesteps_diff


def normalize(x):
    return x
    # return x / x.abs().max(dim=0)[0][None, ...]


def sample_euler(bridge, model, noise, params, save_history=False):
    num_steps = params["num_steps"]
    if params["timesteps"] == "uniform":
        t_steps = get_timesteps_uniform(params, device=noise.device)
    elif params["timesteps"] == "diffusion":
        t_steps = get_timesteps_diff(params, device=noise.device)
    else:
        raise NotImplementedError
    x = noise
    if save_history:
        vis_steps = params["vis_steps"]
        x_history = [normalize(noise)]
    with torch.no_grad():
        for i in range(len(t_steps) - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            t_net = t_steps[i] * torch.ones(x.shape[0], device=noise.device)
            t_delta = t_next - t_cur
            mean = bridge.velocity(model, x, t_net) * t_delta
            std = bridge.diffusion_coef(t_net) * torch.sqrt(t_delta)
            x = x + mean + std[:, None, None, None] * torch.randn_like(x)
            if save_history:
                x_history.append(normalize(x))
    if save_history:
        x_history = (
            [x_history[0]]
            + x_history[1 : -1 : (num_steps - 2) // (vis_steps - 2)]
            + [x_history[-1]]
        )
        return x, x_history
    return x, []
