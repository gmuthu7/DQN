from typing import Callable

import torch
from torch import Tensor

from policies.base_policy import Policy


class EpsilonPolicy(Policy):

    def __init__(self,
                 epsilon_scheduler: Callable[[int], float],
                 action_sampler: Callable[[], Tensor],  # Mediator pattern
                 wrapper_policy: Policy):
        self.wrapper_policy = wrapper_policy
        self.action_sampler = action_sampler
        self.epsilon_scheduler = epsilon_scheduler

    def choose_action(self, state: Tensor, step: int,
                      callback: Callable) -> Tensor:
        epsilon = self.epsilon_scheduler(step)
        callback({
            "train_epsilon": epsilon
        })
        prob = torch.rand(size=(len(state),))
        sample_actions = self.action_sampler()
        actions = self.wrapper_policy.choose_action(state, step, callback)
        actions[prob < epsilon] = sample_actions[prob < epsilon]
        return actions
