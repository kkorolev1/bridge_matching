import torch
import torch.nn as nn


class BridgeMatchingLoss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, model, x_orig, x_trans):
        t = torch.rand(x_orig.shape[0], device=x_orig.device)
        mean = t[:, None, None, None] * x_orig + (1 - t[:, None, None, None]) * x_trans
        std = torch.sqrt(self.gamma * t * (1 - t))
        x_t = mean + std[:, None, None, None] * torch.randn_like(x_orig)
        pred_vf = model(x_t, t)
        gt_cond_vf = (x_orig - x_t) / (1 - t[:, None, None, None] + 1e-5)
        loss = ((pred_vf - gt_cond_vf) ** 2).mean()

        return {"loss": loss, "pred_vf": pred_vf, "gt_cond_vf": gt_cond_vf}
