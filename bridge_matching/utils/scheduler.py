from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmUpLRWithDecay(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_lr,
        total_steps,
        decay_strategy="exponential",
        decay_rate=1,
        last_epoch=-1,
    ):

        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.decay_rate = decay_rate
        self.decay_strategy = decay_strategy
        assert decay_strategy in ["cosine", "exponential"]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [
                base_lr + warmup_factor * (self.max_lr - base_lr)
                for base_lr in self.base_lrs
            ]
        else:
            decay_step = self.last_epoch - self.warmup_steps
            decay_steps = self.total_steps - self.warmup_steps

            if self.decay_strategy == "cosine":
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
                return [self.max_lr * cosine_decay for _ in self.base_lrs]
            elif self.decay_strategy == "exponential":

                exponential_decay = self.max_lr * (
                    self.decay_rate ** (decay_step / decay_steps)
                )
                return [exponential_decay for _ in self.base_lrs]
