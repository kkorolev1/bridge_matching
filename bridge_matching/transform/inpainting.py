class InpaintingTransform:
    def __call__(self, x0):
        x1 = x0.detach().clone()
        image_res = x0.shape[-1]
        x1[..., : image_res // 2, : image_res // 2] = 0
        return x1
