from itertools import repeat
from PIL import Image
import torch
import pandas as pd
import copy


def inf_loop(dataloader):
    """wrapper function for endless data loader."""
    for loader in repeat(dataloader):
        yield from loader


def tensor_to_image(tensor):
    return Image.fromarray(
        (tensor * 127.5 + 127.5)
        .clip(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .detach()
        .numpy()
    )


def copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
