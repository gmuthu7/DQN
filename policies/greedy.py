from typing import Callable

import torch
from torch import Tensor

from policies.base_policy import Policy


class GreedyPolicy(Policy):

    def __init__(self, q_fn: Callable[[Tensor], Tensor]):
        self.q_fn = q_fn

    def choose_action(self, state: Tensor, step: int, callback: Callable) -> Tensor:
        q_values = self.q_fn(state)
        return torch.argmax(q_values, dim=-1)
