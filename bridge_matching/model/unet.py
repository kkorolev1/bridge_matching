import torch.nn as nn


class DummyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
