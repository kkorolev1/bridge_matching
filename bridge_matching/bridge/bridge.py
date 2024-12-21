import torch
import math
from .schedule import Schedule


class Bridge:
    def sample_t(self, x_orig, x_trans, t):
        """Samples x_t from x_0=x_trans and x_1=x_orig"""
        raise NotImplementedError

    def velocity(self, model, x, t):
        model_pred = model(x, t)
        if model.predict_type == "x_orig":
            return self.to_velocity(model_pred, x, t)
        elif model.predict_type == "velocity":
            return model_pred

    def to_velocity(self, x_orig, x, t):
        raise NotImplementedError

    def diffusion_coef(self, t):
        """Diffusion for the SDE"""
        raise NotImplementedError


class GeneralizedBrownianBridge(Bridge):
    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def x_orig_mean_coef(self, t):
        """Coefficient in front of x_1 in x_t mean
        x_orig is x_1 from the paper
        """
        return self.schedule.sigma2_t(t) / self.schedule.sigma2_sum_t(t)

    def x_trans_mean_coef(self, t):
        """Coefficient in front of x_0 in x_t mean
        x_trans is x_0 from the paper
        """
        return self.schedule.sigma2_bar_t(t) / self.schedule.sigma2_sum_t(t)

    def var_t(self, t):
        """Variance of the x_t"""
        return (
            self.schedule.sigma2_t(t)
            * self.schedule.sigma2_bar_t(t)
            / self.schedule.sigma2_sum_t(t)
        )

    def sample_t(self, x_orig, x_trans, t):
        mean = (
            self.x_orig_mean_coef(t)[:, None, None, None] * x_orig
            + self.x_trans_mean_coef(t)[:, None, None, None] * x_trans
        )
        std = torch.sqrt(self.var_t(t))
        x_t = mean + std[:, None, None, None] * torch.randn_like(x_orig)
        return x_t

    def to_velocity(self, x_orig, x, t):
        eps = 1e-4
        return (
            self.schedule.beta_t(t)[:, None, None, None]
            * (x_orig - x)
            / (self.schedule.sigma2_bar_t(t)[:, None, None, None] + eps)
        )

    def diffusion_coef(self, t):
        return torch.sqrt(self.schedule.beta_t(t))


class BrownianBridge(Bridge):
    def __init__(self, gamma):
        self.gamma = gamma

    def sample_t(self, x_orig, x_trans, t):
        mean = t[:, None, None, None] * x_orig + (1 - t[:, None, None, None]) * x_trans
        std = torch.sqrt(self.gamma * t * (1 - t))
        x_t = mean + std[:, None, None, None] * torch.randn_like(x_orig)
        return x_t

    def to_velocity(self, x_orig, x, t):
        eps = 1e-4
        return (x_orig - x) / (1 - t[:, None, None, None] + eps)

    def diffusion_coef(self, t):
        return math.sqrt(self.gamma) * torch.ones_like(t)


class FlowMatching(Bridge):
    def sample_t(self, x_orig, x_trans, t):
        mean = t[:, None, None, None] * x_orig + (1 - t[:, None, None, None]) * x_trans
        return mean

    def to_velocity(self, x_orig, x, t):
        eps = 1e-4
        return (x_orig - x) / (1 - t[:, None, None, None] + eps)

    def diffusion_coef(self, t):
        return torch.zeros_like(t)
