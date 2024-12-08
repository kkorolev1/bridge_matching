import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from bridge_matching.utils import inf_loop


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        config,
        dataloaders,
        device,
        accelerator,
        transform,
        logger,
        skip_oom=True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.transform = transform
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.train_dataloader = inf_loop(self.train_dataloader)
        self.len_epoch = config.trainer.len_epoch
        self.device = device
        self.logger = logger
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.log_step = config.trainer.log_step
        self.start_epoch = 1
        self.epochs = config.trainer.epochs
        self.accelerator = accelerator
        if config.resume:
            self._resume_checkpoint(config.resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            with self.accelerator.accumulate(self.model):  # grad accum
                try:
                    batch = self.train_step(batch, batch_idx)
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            if batch_idx % self.log_step == 0:
                if self.accelerator.is_main_process:
                    self.logger.info(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"]
                        )
                    )
                log_step = {
                    "epoch": epoch,
                    "loss": batch["loss"],
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                self.accelerator.log(
                    log_step, step=self.get_global_step(epoch, batch_idx)
                )

            if batch_idx + 1 >= self.len_epoch:
                break

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)

    def get_global_step(self, epoch, batch_idx):
        return (epoch - 1) * self.len_epoch + batch_idx

    def train_step(self, batch, batch_idx):
        x0 = batch
        x1 = self.transform(x0)
        loss_dict = self.criterion(self.model, x0, x1)
        self.accelerator.backward(loss_dict["loss"])
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.grad_norm_clip
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return {"loss": loss_dict["loss"].item()}

    @torch.no_grad()
    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    batch_idx,
                )

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def __save_model(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(self.model, self.config.trainer.save_dir)
