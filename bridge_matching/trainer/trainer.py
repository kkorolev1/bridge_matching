import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import os
import shutil
import dill
import threading
from hydra.core.hydra_config import HydraConfig
from torchvision.utils import make_grid
from bridge_matching.utils import inf_loop, tensor_to_image, copy_to_cpu, MetricTracker
from bridge_matching.sampler import sample_euler


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
        bridge,
        sampling_params,
        metrics,
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
        self.save_epoch = config.trainer.save_epoch
        self.epoch = 1
        self.epochs = config.trainer.epochs
        self.accelerator = accelerator
        self.bridge = bridge
        self.sampling_params = sampling_params
        self.metrics = metrics
        self.train_metric_tracker = MetricTracker("loss", "grad_norm")
        self.output_dir = (
            config.trainer.output_dir
            if hasattr(config.trainer, "output_dir")
            else HydraConfig.get().runtime.output_dir
        )
        if config.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                self.logger.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

    def train(self):
        while self.epoch < self.epochs + 1:
            self._train_epoch(self.epoch)
            self.epoch += 1

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metric_tracker.reset()
        self.accelerator.log({"epoch": epoch}, step=self._global_step(epoch, 0))

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            with self.accelerator.accumulate(self.model):  # grad accum
                try:
                    step_dict = self.train_step(
                        batch, batch_idx, self.train_metric_tracker
                    )
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
                            epoch, self._progress(batch_idx), step_dict["loss"]
                        )
                    )
                self._log_scalars(self.train_metric_tracker, epoch, batch_idx)
                self.accelerator.log(
                    {"lr": self.lr_scheduler.get_last_lr()[0]},
                    step=self._global_step(epoch, batch_idx),
                )
                self._log_train_batch(
                    step_dict, step=self._global_step(epoch, batch_idx)
                )
                self.train_metric_tracker.reset()

            if (
                batch_idx % self.config.trainer.viz_step == 0
                and self.accelerator.is_main_process
            ):
                model = self.accelerator.unwrap_model(self.model)
                self._log_predictions(
                    model,
                    batch,
                    step=self._global_step(epoch, batch_idx),
                    label="train",
                )

            if batch_idx + 1 >= self.len_epoch:
                break

        if self.accelerator.is_main_process:
            for part, dataloader in self.evaluation_dataloaders.items():
                self._evaluation_epoch(epoch, part, dataloader)

        if (epoch - 1) % self.save_epoch == 0 and self.accelerator.is_main_process:
            self.save_checkpoint()

    def _global_step(self, epoch, batch_idx):
        return (epoch - 1) * self.len_epoch + batch_idx

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def train_step(self, batch, batch_idx, metric_tracker):
        x_orig = batch
        x_trans = self.transform(x_orig)
        loss_dict = self.criterion(self.bridge, self.model, x_orig, x_trans)
        self.accelerator.backward(loss_dict["loss"])
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.grad_norm_clip
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        result_dict = {
            "x_orig": x_orig,
            "x_trans": x_trans,
            "grad_norm": self.get_grad_norm(self.model),
        }
        self.optimizer.zero_grad()
        result_dict.update(loss_dict)

        for metric_key in metric_tracker.keys():
            value = result_dict[metric_key]
            metric_tracker.update(
                metric_key, value.item() if isinstance(value, torch.Tensor) else value
            )

        return result_dict

    @torch.no_grad()
    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        image_index = 1
        predictions_dir = (
            Path(HydraConfig.get().runtime.output_dir)
            / self.config.trainer.predictions_dir
        )
        if os.path.exists(predictions_dir):
            shutil.rmtree(predictions_dir)
        os.makedirs(predictions_dir, exist_ok=True)
        fid_metric = self.metrics[0]  # hardcode it for now
        model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                if image_index >= fid_metric.num_expected:
                    break
                x_orig = batch
                x_trans = self.transform(x_orig)
                x_pred, _ = sample_euler(
                    self.bridge,
                    model,
                    x_trans,
                    self.sampling_params,
                    save_history=False,
                )
                for i in range(x_pred.shape[0]):
                    image = tensor_to_image(x_pred[i])
                    image.save(predictions_dir / f"{image_index}.png")
                    image_index += 1
        fid = fid_metric(predictions_dir)
        self.accelerator.log(
            {"FID": fid},
            step=self._global_step(epoch, self.len_epoch),
        )
        self._log_predictions(
            model, batch, step=self._global_step(epoch, self.len_epoch), label="test"
        )
        if os.path.exists(predictions_dir):
            shutil.rmtree(predictions_dir)

    def _log_scalars(self, metric_tracker: MetricTracker, epoch, batch_idx):
        for metric_name in metric_tracker.keys():
            self.accelerator.log(
                {f"{metric_name}": metric_tracker.avg(metric_name)},
                step=self._global_step(epoch, batch_idx),
            )

    def _log_train_batch(self, step_dict, step, n_pictures_sampling=8):
        images = torch.cat(
            [
                step_dict["x_orig"][:n_pictures_sampling],
                step_dict["x_trans"][:n_pictures_sampling],
                step_dict["x_t"][:n_pictures_sampling],
                step_dict["pred"][:n_pictures_sampling],
                step_dict["gt"][:n_pictures_sampling],
            ],
            dim=0,
        )
        image_grid = make_grid(images, nrow=n_pictures_sampling)
        self.accelerator.trackers[0].log_images(
            {"train_grid": [tensor_to_image(image_grid)]}, step=step
        )

    def _log_predictions(
        self, model, x_orig, step, label="train", n_pictures_sampling=8
    ):
        x_trans = self.transform(x_orig)
        x_pred, trajectory = sample_euler(
            self.bridge, model, x_trans, self.sampling_params, save_history=True
        )
        images = torch.cat(
            [
                x_orig[:n_pictures_sampling],
                x_trans[:n_pictures_sampling],
                x_pred[:n_pictures_sampling],
            ],
            dim=0,
        )
        image_grid = make_grid(images, nrow=n_pictures_sampling)
        self.accelerator.trackers[0].log_images(
            {f"{label}_predictions": [tensor_to_image(image_grid)]},
            step=step,
        )

        trajectory_len = len(trajectory)
        trajectory = torch.stack(trajectory, dim=0)
        trajectory = trajectory[:, :n_pictures_sampling, ...]
        trajectory = trajectory.permute(1, 0, 2, 3, 4).reshape(
            -1, *trajectory.shape[-3:]
        )
        trajectory_grid = make_grid(trajectory, nrow=trajectory_len)
        self.accelerator.trackers[0].log_images(
            {f"{label}_trajectory": [tensor_to_image(trajectory_grid)]},
            step=step,
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

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = Path(self.output_dir).joinpath(
                "checkpoints", f"{self.config.logging.name}_{tag}.ckpt"
            )
        else:
            path = Path(path)
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = tuple() + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"config": self.config, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload
