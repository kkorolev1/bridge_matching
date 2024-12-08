import warnings
import sys

import numpy as np
import torch


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator

from bridge_matching.trainer import Trainer
from bridge_matching.dataset.dataloader import get_dataloaders

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
    accelerator = instantiate(config.accelerate)
    wandb_config = OmegaConf.to_container(config.logging, resolve=True)
    # wandb_config.pop("project")
    # accelerator.init_trackers(
    #     project_name=config.logging.project,
    #     config=OmegaConf.to_container(config, resolve=True),
    #     init_kwargs={"wandb": wandb_config},
    # )

    dataloaders = get_dataloaders(config.data)

    model = instantiate(config.model)

    loss_module = instantiate(config.loss)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    for k, v in zip(dataloaders.keys(), accelerator.prepare(*dataloaders.values())):
        dataloaders[k] = v
    device = model.device

    accelerator.register_for_checkpointing(lr_scheduler)

    trainer = Trainer(
        model,
        loss_module,
        optimizer,
        lr_scheduler,
        config=config,
        dataloaders=dataloaders,
        device=device,
        accelerator=accelerator
    )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    main()
