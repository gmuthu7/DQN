import torch
from torch import Tensor

from agents.dqn import Dqn


class DoubleDqn(Dqn):

    def _get_target(self, reward: Tensor, next_state: Tensor, terminated: Tensor, gamma: float) -> Tensor:
        with torch.no_grad():
            batch_size = next_state.shape[0]
            target_output = self.target_vfa.val(next_state)
            action = self.vfa_greedy_policy.choose_action(next_state, 0, lambda: None)
            val_action = target_output[torch.arange(batch_size), action]
            return torch.where(terminated, reward, reward + gamma * val_action)
