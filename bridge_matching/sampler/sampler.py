import torch.nn as nn
import torch

def normalize(x):
    return x / x.abs().max(dim=0)[0][None, ...]

def get_timesteps():
    pass 

def sample_euler(model, noise, params, gamma=1, **model_kwargs):
    t_steps = get_timesteps(params)
    x = noise
    with torch.no_grad():
        for i in range(len(t_steps) - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            t_net = t_steps[i] * torch.ones(x.shape[0], device=model.device)
            # check formula? 
            x = x + model(x, t_net) * (t_next - t_cur) + torch.randn_like(x) * torch.sqrt(gamma) * torch.sqrt(abs(t_next - t_cur))
    return x