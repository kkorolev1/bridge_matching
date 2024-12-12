import torch
import torch.nn as nn


class BridgeMatchingLoss:
    def __call__(self, bridge, model, x_orig, x_trans):
        eps = 1e-5
        t = torch.rand(x_orig.shape[0], device=x_orig.device)
        x_t = bridge.sample_t(x_orig, x_trans, t)
        pred_vf = model(x_t, t)
        gt_vf = (x_orig - x_t) / (1 - t[:, None, None, None] + eps)
        loss = ((pred_vf - gt_vf) ** 2).mean()
        return {"loss": loss, "pred_vf": pred_vf, "gt_vf": gt_vf}
