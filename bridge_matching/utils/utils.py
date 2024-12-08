from itertools import repeat
from PIL import Image
import torch


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
