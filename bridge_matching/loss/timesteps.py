import torch


class UniformSampler:
    def __call__(self, batch_size, device):
        return torch.rand(batch_size, device=device)


class LogitSampler:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, batch_size, device):
        return torch.sigmoid(
            self.mean + self.std * torch.randn(batch_size, device=device)
        )
