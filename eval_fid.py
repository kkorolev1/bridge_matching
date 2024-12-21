import warnings
import sys
from tqdm.auto import trange
import os
import numpy as np
import torch
from pathlib import Path
import shutil
import hydra
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import dill

from bridge_matching.dataset.dataloader import get_dataloaders
from bridge_matching.utils import tensor_to_image
from bridge_matching.sampler import sample_euler

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def load_state_dict(path):
    payload = torch.load(Path(path).open("rb"), pickle_module=dill)
    state_dict = payload["state_dicts"]["model"]
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("module"):
            new_state_dict[key[7:]] = state_dict[key]
    state_dict = new_state_dict
    return state_dict


@hydra.main(version_base=None, config_path="bridge_matching/config")
@torch.inference_mode
def main(config: DictConfig):
    OmegaConf.resolve(config)
    checkpoint_path = Path(config.checkpoint_path)
    state_dict = load_state_dict(checkpoint_path)
    model = instantiate(config.model)
    model.load_state_dict(state_dict)
    model = model.cuda()
    bridge = instantiate(config.bridge)
    transform = instantiate(config.transform)
    sampling_params = config.sampling_params
    fid_metric = instantiate(config.metrics[0])

    images_dir = Path(config.images_dir)
    dataloader = get_dataloaders(config.data)["test"]

    model.eval()
    image_index = 1
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Calculating FID",
            total=len(dataloader),
        ):
            x_orig = batch.cuda()
            x_trans = transform(x_orig)
            x_pred, _ = sample_euler(
                bridge,
                model,
                x_trans,
                sampling_params,
                save_history=False,
            )
            for i in range(x_pred.shape[0]):
                image = tensor_to_image(x_pred[i])
                image.save(images_dir / f"{image_index}.png")
                image_index += 1
        fid = fid_metric(images_dir)
        print(f"FID: {fid}")

        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    main()
