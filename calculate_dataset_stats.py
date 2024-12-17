import warnings
import sys
from tqdm.auto import trange
import os
import numpy as np
import torch
from pathlib import Path
import pickle

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bridge_matching.utils import tensor_to_image
from bridge_matching.metric.fid import calculate_inception_stats

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="bridge_matching/config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    dataset = instantiate(config.dataset)
    images_dir = Path(config.images_dir)
    output_dir = Path(config.output_path)

    if not images_dir.exists():
        os.makedirs(images_dir, exist_ok=True)
    if not output_dir.parent.exists():
        os.makedirs(output_dir.parent, exist_ok=True)

    for i in trange(len(dataset)):
        tensor_to_image(dataset[i]).save(images_dir / f"{i + 1}.png")

    mu, sigma = calculate_inception_stats(
        images_dir, num_expected=len(dataset), max_batch_size=config.batch_size
    )
    result_dict = {"mu": mu, "sigma": sigma}
    with open(config.output_path, "wb") as f:
        pickle.dump(result_dict, f)
    print(f"Saved statistics to {config.output_path}")


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    main()
