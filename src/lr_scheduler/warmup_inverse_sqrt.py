from torch.optim.lr_scheduler import LambdaLR


class WarmupInverseSquareRootLR(LambdaLR):
    """Warm up to optimizer.lr, then decay by the inverse square root of the step."""

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch=last_epoch)

    def _lr_multiplier(self, step):
        step += 1
        return min(step / self.warmup_steps, (self.warmup_steps / step) ** 0.5)
