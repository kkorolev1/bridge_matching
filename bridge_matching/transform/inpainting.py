import torch


class InpaintingTransform:
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def __call__(self, x_orig, random_mask=False, **kwargs):
        x_trans = x_orig.detach().clone()
        image_res = x_trans.shape[-1]
        if self.noise_std > 0:
            value = self.noise_std * torch.randn_like(x_trans)
        else:
            value = torch.zeros_like(x_trans)
        if random_mask:
            width, height = torch.randint(
                low=image_res // 2, high=3 * image_res // 4, size=(2,)
            )
            x, y = torch.randint(low=0, high=image_res // 2, size=(2,))
            x_trans[..., y : y + height, x : x + width] = value[
                ..., y : y + height, x : x + width
            ]
        else:
            x_trans[..., : image_res // 2, :] = value[..., : image_res // 2, :]
        return x_trans
