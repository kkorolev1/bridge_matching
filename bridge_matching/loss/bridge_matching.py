import torch
import torch.nn as nn


class BridgeMatchingLoss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, model, x0, x1):
        t = torch.rand(x0.shape[0], device=x0.device)
        mean = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
        std = torch.sqrt(self.gamma * t * (1 - t))
        xt = mean + std[:, None, None, None] * torch.randn_like(x0)
        pred_vel = model(xt, t)
        gt_cond_vel = (x1 - xt) / (1 - t[:, None, None, None])
        loss = ((pred_vel - gt_cond_vel) ** 2).mean()

        return {"loss": loss, "pred_vel": pred_vel, "gt_cond_vel": gt_cond_vel}
