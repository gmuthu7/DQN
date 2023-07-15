import copy
from typing import Callable

import torch
from numpy import ndarray
from torch import Tensor

from agents.base_agent import Agent
from buffers.experience_replay import ExperienceReplay
from policies.base_policy import Policy
from policies.greedy import GreedyPolicy
from vfa.neural_network import NeuralNetworkVfa


class Dqn(Agent):

    def __init__(self,
                 buffer: ExperienceReplay,
                 vfa: NeuralNetworkVfa,
                 behaviour_policy: Policy,
                 update_freq: int,
                 target_update_freq: int,
                 initial_no_learn_steps: int,
                 ):
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.vfa = vfa
        self.behaviour_policy = behaviour_policy
        self.target_vfa = copy.deepcopy(vfa)
        self.initial_no_learn_steps = initial_no_learn_steps
        self.buffer = buffer
        self.vfa_greedy_policy = GreedyPolicy(vfa.val)
        self.target_greedy_policy = GreedyPolicy(self.target_vfa.val)

    def learn(self,
              state: ndarray,
              action: ndarray,
              next_state: ndarray,
              reward: ndarray,
              terminated: ndarray,
              truncated: ndarray,
              gamma: float,
              step: int,
              callback: Callable):
        self.buffer.store(state, action, next_state, reward, terminated)
        if step % self.update_freq == 0:
            state, action, next_state, reward, terminated = self.buffer.sample()
            target = self._get_target(reward, next_state, terminated, gamma)
            pred = self.vfa.val(state)
            pred = torch.gather(pred, 1, action.unsqueeze(1)).flatten()
            self.vfa.step(pred, target, callback)
        if step % self.target_update_freq == 0:
            self.target_vfa = copy.deepcopy(self.vfa)

    def act_to_learn(self, state: ndarray, step: int, callback: Callable) -> ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state)
            actions = self.behaviour_policy.choose_action(state, step, callback)
            return actions.cpu().numpy()

    def act(self, state: ndarray) -> ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state)
            actions = self.vfa_greedy_policy.choose_action(state, 0, lambda: None)
            return actions.cpu().numpy()

    def _get_target(self, reward: Tensor, next_state: Tensor, terminated: Tensor, gamma: float) -> Tensor:
        with torch.no_grad():
            target_output = self.target_vfa.val(next_state)
            action = self.target_greedy_policy.choose_action(next_state, 0, lambda: None).unsqueeze(1)
            return reward + (1 - terminated) * gamma * torch.gather(target_output, 1, action).flatten()
