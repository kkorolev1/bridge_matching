import torch


class Schedule:
    def beta_t(self, t):
        """beta_t from the paper"""
        raise NotImplementedError

    def sigma2_t(self, t):
        """sigma^2_t from the paper"""
        raise NotImplementedError

    def sigma2_bar_t(self, t):
        """bar{sigma}^2_t from the paper"""
        raise NotImplementedError

    def sigma2_sum_t(self, t):
        """sigma^2_t + bar{sigma}^2_t from the paper"""
        raise NotImplementedError


class TriangularSchedule(Schedule):
    def __init__(self, beta_max):
        self.beta_max = beta_max

    def beta_t(self, t):
        return 2 * self.beta_max * torch.min(t, (1 - t))

    def sigma2_t(self, t):
        s2_t = torch.empty_like(t)
        s2_t[t <= 0.5] = self.beta_max * (t**2)[t <= 0.5]
        s2_t[t >= 0.5] = self.beta_max * (0.5 - (1 - t) ** 2)[t >= 0.5]
        return s2_t

    def sigma2_bar_t(self, t):
        s2_t = torch.empty_like(t)
        s2_t[t <= 0.5] = self.beta_max * (0.5 - t**2)[t <= 0.5]
        s2_t[t >= 0.5] = self.beta_max * ((1 - t) ** 2)[t >= 0.5]
        return s2_t

    def sigma2_sum_t(self, t):
        return 0.5 * self.beta_max * torch.ones_like(t)
