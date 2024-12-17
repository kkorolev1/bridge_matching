import torch


class InpaintingTransform:
    def __call__(self, x_orig, random=False, **kwargs):
        x_trans = x_orig.detach().clone()
        image_res = x_trans.shape[-1]
        if random:
            width, height = torch.randint(
                low=image_res // 2, high=3 * image_res // 4, size=(2,)
            )
            x, y = torch.randint(low=0, high=image_res // 2, size=(2,))
            x_trans[..., y : y + height, x : x + width] = 0
        else:
            x_trans[..., : image_res // 2, :] = 0
        return x_trans
