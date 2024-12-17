import torch
import torch.nn as nn
from edm.training.networks import SongUNet, DhariwalUNet


class BridgeMatchingModel(nn.Module):
    def __init__(self, img_resolution=64, predict_type="x_orig"):
        super().__init__()
        assert predict_type in [
            "x_orig",
            "velocity",
        ], f"Unknown predict type {predict_type}"
        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=3,
            out_channels=3,
        )
        self.predict_type = predict_type

    def forward(self, x, t):
        class_labels = torch.zeros_like(t)
        return self.model(x=x, noise_labels=t, class_labels=class_labels)

    def to_velocity(self, model_pred, x, t):
        eps = 1e-5
        if self.predict_type == "x_orig":
            return (model_pred - x) / (1 - t[:, None, None, None] + eps)
        elif self.predict_type == "velocity":
            return model_pred

    def velocity(self, x, t):
        return self.to_velocity(self.forward(x, t), x, t)
