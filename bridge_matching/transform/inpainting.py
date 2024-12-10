class InpaintingTransform:
    def __call__(self, x_orig):
        x_trans = x_orig.detach().clone()
        image_res = x_trans.shape[-1]
        x_trans[..., : image_res // 2, :] = 0
        return x_trans
