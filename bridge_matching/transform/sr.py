import torch
from torchvision.transforms import v2


class SuperResolutionTransform:
    def __init__(self, factor=4):
        self.factor = factor

    def __call__(self, x_orig, **kwargs):
        image_res = x_orig.shape[-1]
        trans_res = image_res // self.factor
        return v2.functional.resize(
            v2.functional.resize(x_orig, (trans_res, trans_res)),
            (image_res, image_res),
        )
