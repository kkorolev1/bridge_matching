import warnings
import sys
import os
import logging

import numpy as np
import torch

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
import wandb

from bridge_matching.trainer import Trainer
from bridge_matching.dataset.dataloader import get_dataloaders
from bridge_matching.logger import setup_logging

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
    accelerator = Accelerator(log_with="wandb")

    logger = logging.getLogger("train")
    setup_logging("logs")

    wandb_config = OmegaConf.to_container(config.logging, resolve=True)
    if not config.logging.debug:
        wandb_config.pop("project")
        wandb_config.pop("debug")

        wandb.login(relogin=True, key=os.environ["WANDB_API_KEY"])

        accelerator.init_trackers(
            project_name=config.logging.project,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": wandb_config},
        )
    dataloaders = get_dataloaders(config.data)

    model = instantiate(config.model)

    loss_module = instantiate(config.loss)
    transform = instantiate(config.transform)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    for k, v in zip(dataloaders.keys(), accelerator.prepare(*dataloaders.values())):
        dataloaders[k] = v
    device = model.device

    if accelerator.is_main_process:
        num_params = sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        )
        logger.info(f"Trainable parameters {num_params}")

    bridge = instantiate(config.bridge)
    sampling_params = OmegaConf.to_container(config.sampling_params, resolve=True)

    metrics = [instantiate(metric_dict) for metric_dict in config.metrics]

    trainer = Trainer(
        model,
        loss_module,
        optimizer,
        lr_scheduler,
        config=config,
        dataloaders=dataloaders,
        device=device,
        accelerator=accelerator,
        transform=transform,
        logger=logger,
        bridge=bridge,
        sampling_params=sampling_params,
        metrics=metrics,
    )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    main()
