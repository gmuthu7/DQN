from typing import Dict, Callable

import torch
import torch.nn as nn
from torch import Tensor


class NeuralNetworkVfa:

    def __init__(self,
                 network: nn.Module,
                 loss_fn: torch.nn.modules.loss,
                 optimizer: torch.optim.Optimizer,
                 clip_val: float):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_val = clip_val
        self.network = network

    def val(self, x: Tensor) -> Tensor:
        logits = self.network(x)
        return logits

    def step(self, pred: Tensor, target: Tensor,
             callback: Callable[[Dict], None]):
        loss = self.loss_fn(target, pred)
        self.optimizer.zero_grad()
        loss.backward()
        before_norm = self._get_grad_norm()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_val)
        after__norm = self._get_grad_norm()
        self.optimizer.step()
        callback({"train_mean_loss": loss,
                  "train_mean_qfn": torch.mean(pred),
                  "train_before_clip_grad": before_norm,
                  "train_after_clip_grad": after__norm,
                  })

    def _get_grad_norm(self) -> float:
        grads = []
        for param in self.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return torch.linalg.norm(grads)
