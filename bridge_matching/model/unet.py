import torch
import torch.nn as nn
from edm.training.networks import SongUNet


class UNet(nn.Module):
    def __init__(self, img_resolution=64):
        super().__init__()
        self.model = SongUNet(
            img_resolution=img_resolution, in_channels=3, out_channels=3
        )

    def forward(self, x, t):
        class_labels = torch.zeros_like(t)
        return self.model(x=x, noise_labels=t, class_labels=class_labels)
