import torch


def normalize(x):
    return x / x.abs().max(dim=0)[0][None, ...]


def get_timesteps(params, device):
    num_steps = params["num_steps"]
    return torch.linspace(1e-4, 1, num_steps + 1, device=device)


def sample_euler(model, noise, params, save_history=False):
    num_steps = params["num_steps"]
    t_steps = get_timesteps(params, device=noise.device)
    gamma = params["gamma"]
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
            model_pred = model(x, t_net)
            x = (
                x
                + model_pred * t_delta
                + torch.randn_like(x) * torch.sqrt(gamma * t_delta)
            )
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
