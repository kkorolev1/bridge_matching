import numpy as np
import pickle
from .base import BaseMetric
from .fid_utils import calculate_fid_from_inception_stats, calculate_inception_stats


class FIDMetric(BaseMetric):
    def __init__(self, name, ref_path, num_expected, batch_size):
        super().__init__(name)
        self.ref_path = ref_path
        self.num_expected = num_expected
        self.batch_size = batch_size

    def __call__(self, image_path, **kwargs):
        with open(self.ref_path, "rb") as f:
            ref = pickle.load(f)
        mu, sigma = calculate_inception_stats(
            image_path=image_path,
            num_expected=self.num_expected,
            max_batch_size=self.batch_size,
        )
        fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
        return fid
