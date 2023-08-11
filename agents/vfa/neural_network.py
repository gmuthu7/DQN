from typing import Dict, Callable, Self

import torch
import torch.nn as nn
from torch import Tensor


class NeuralNetworkVfa:

    def __init__(self,
                 network: nn.Module,
                 loss_fn: torch.nn.modules.loss,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                 clip_val: float):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_val = clip_val
        self.network = network
        self.scheduler = scheduler

    def val(self, x: Tensor) -> Tensor:
        logits = self.network(x)
        return logits

    def step(self, pred: Tensor, target: Tensor,
             callback: Callable[[Dict], None]):
        callback_dict = {}
        loss = self.loss_fn(target, pred)
        self.optimizer.zero_grad()
        loss.backward()
        before_norm = self._get_grad_norm()
        self._clip_grad(callback_dict)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            callback_dict.update({"train/lr": self.scheduler.get_last_lr()})
        callback_dict.update({"train/mean_loss": loss,
                              "train/mean_qfn": torch.mean(pred),
                              "train/before_clip_grad": before_norm,
                              })
        callback(callback_dict)

    def clone(self, vfa: Self):
        self.network.load_state_dict(vfa.network.state_dict())

    def _clip_grad(self, callback_dict: Dict):
        if self.clip_val > 0.:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_val)
            after_norm = self._get_grad_norm()
            callback_dict.update({
                "train/after_clip_grad": after_norm,
            })

    def _get_grad_norm(self) -> float:
        grads = []
        for param in self.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return torch.linalg.norm(grads)
