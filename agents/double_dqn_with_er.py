import torch

from agents.dqn_with_er import DqnWithExperienceReplay


class DoubleDqnWithExperienceReplay(DqnWithExperienceReplay):

    def _get_target(self, breward: torch.Tensor, bnext_state: torch.Tensor, bdone: torch.Tensor) -> torch.Tensor:
        target_output = self.target_network(bnext_state)
        return breward + (1 - bdone) * self.gamma * target_output[
            torch.arange(len(target_output)), self._greedy_action(self.q_network, bnext_state)]
