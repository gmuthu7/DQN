import torch
from torch import Tensor

from agents.dqn import Dqn


class DoubleDqn(Dqn):

    def _get_target(self, reward: Tensor, next_state: Tensor, terminated: Tensor, gamma: float) -> Tensor:
        with torch.no_grad():
            target_output = self.target_vfa.val(next_state)
            action = self.vfa_greedy_policy.choose_action(next_state, 0, lambda: None).unsqueeze(1)
            return reward + (1 - terminated) * gamma * torch.gather(target_output, 1, action).squeeze()
