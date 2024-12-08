import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from accelerate import Accelerator
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
        skip_oom=True,
        accelerator=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.train_dataloader = inf_loop(self.train_dataloader)
        self.len_epoch = config.trainer.len_epoch
        self.device = device
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.log_step = config.trainer.log_step
        self.start_epoch = 1
        self.epochs = config.trainer.epochs
        self.accelerator = accelerator
        if config.resume:
            self._resume_checkpoint(config.resume)

        self.accelerator.init_trackers(
            "BMProject",
            config={"Loss": None,
            "epoch": self.start_epoch,
            "lr": config.optimizer.lr}
        )


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
        self.accelerator.log({"epoch" : epoch})

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            with self.accelerator.accumulate(self.model): # grad accum
                try:
                    batch = self.process_batch(batch, batch_idx)
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
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.accelerator.log(
                    {"lr", self.lr_scheduler.get_last_lr()[0]}
                ) # TODO: Add logging of other things


            if batch_idx + 1 >= self.len_epoch:
                break

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)

    def process_batch(self, batch, batch_idx):

        # TODO: Add self.accelerator.backward(loss), mixed prec 
        if self.accelerator.sync_gradients: 
            self.accelerator.clip_grad_norm_(self.model.parameters(),
                                                self.config["trainer"]["grad_norm_clip"])
            
        return batch

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
        self.accelerator.save_model(self.model, self.config["save_dir"])
