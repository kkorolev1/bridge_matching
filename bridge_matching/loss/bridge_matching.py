import torch
import torch.nn as nn


class BridgeMatchingLoss:
    def __init__(self, timestep_sampler):
        self.timestep_sampler = timestep_sampler

    def __call__(self, bridge, model, x_orig, x_trans):
        t = self.timestep_sampler(x_orig.shape[0], x_orig.device)
        x_t = bridge.sample_t(x_orig, x_trans, t)
        pred = model(x_t, t)
        if model.module.predict_type == "x_orig":
            gt = x_orig
        elif model.module.predict_type == "velocity":
            gt = self.bridge.to_velocity(x_orig, x_t, t)

        loss = ((pred - gt) ** 2).mean()
        return {"loss": loss, "pred": pred, "gt": gt, "x_t": x_t}
