import copy
from typing import Callable

import torch
from numpy import ndarray
from torch import Tensor, arange

from agents.base_agent import Agent
from agents.buffers.experience_replay import ExperienceReplay
from policies.base_policy import Policy
from policies.greedy import GreedyPolicy
from agents.vfa import NeuralNetworkVfa


class Dqn(Agent):

    def __init__(self,
                 buffer: ExperienceReplay,
                 vfa: NeuralNetworkVfa,
                 behaviour_policy: Policy,
                 update_freq: int,
                 target_update_freq: int,
                 initial_no_learn_steps: int,
                 num_update_steps: int
                 ):
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.vfa = vfa
        self.behaviour_policy = behaviour_policy
        self.target_vfa = copy.deepcopy(vfa)
        self.initial_no_learn_steps = initial_no_learn_steps
        self.buffer = buffer
        self.vfa_greedy_policy = GreedyPolicy(self.vfa.val)
        self.target_vfa_greedy_policy = GreedyPolicy(self.target_vfa.val)
        self.num_update_steps = num_update_steps
        self.last_target_update = 1
        self.last_update = 1

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
        if step < self.initial_no_learn_steps:
            return
        if (step - self.last_update) >= self.update_freq:
            self.last_update = step
            for i in range(self.num_update_steps):
                state, action, next_state, reward, terminated = self.buffer.sample()
                batch_size = state.shape[0]
                target = self._get_target(reward, next_state, terminated, gamma)
                pred = self.vfa.val(state)
                pred = pred[arange(batch_size), action]
                self.vfa.step(pred, target, callback if i == 0 else lambda x: None)
        if (step - self.last_target_update) >= self.target_update_freq:
            self.last_target_update = step
            self.target_vfa.clone(self.vfa)

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
            batch_size = next_state.shape[0]
            target_output = self.target_vfa.val(next_state)
            action = self.target_vfa_greedy_policy.choose_action(next_state, 0, lambda: None)
            val_action = target_output[torch.arange(batch_size), action]
            return torch.where(terminated, reward, reward + gamma * val_action)
