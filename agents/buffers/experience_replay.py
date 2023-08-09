from typing import Tuple

import torch
from numpy import ndarray


class ExperienceReplay:

    def __init__(self, buffer_size: int, batch_size: int):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dtype = torch.float
        self.state, self.action, self.next_state, self.reward, self.terminated = self._get_tensors([], [], [], [], [],
                                                                                                   self.dtype)
        self.full = False

    def store(self, state: ndarray, action: ndarray, next_state: ndarray, reward: ndarray, terminated: ndarray):
        num_envs = state.shape[0]
        state, action, next_state, reward, terminated = self._get_tensors(state, action, next_state, reward, terminated,
                                                                          self.dtype)
        if self.full:
            self.state = torch.cat([self.state[num_envs:], state])
            self.action = torch.cat([self.action[num_envs:], action])
            self.next_state = torch.cat([self.next_state[num_envs:], next_state])
            self.reward = torch.cat([self.reward[num_envs:], reward])
            self.terminated = torch.cat([self.terminated[num_envs:], terminated])
        else:
            self.state = torch.cat([self.state, state])
            self.action = torch.cat([self.action, action])
            self.next_state = torch.cat([self.next_state, next_state])
            self.reward = torch.cat([self.reward, reward])
            self.terminated = torch.cat([self.terminated, terminated])
        if not self.full and self.state.shape[0] >= self.buffer_size:
            self.full = True

    def sample(self) -> Tuple:
        pos = torch.randint(0, self.state.shape[0], (self.batch_size,))
        return self.state[pos], self.action[pos], self.next_state[pos], self.reward[pos], self.terminated[pos]

    def _get_tensors(self, state, action, next_state, reward, terminated, dtype):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=dtype)
            action = torch.as_tensor(action, dtype=torch.int64)
            next_state = torch.as_tensor(next_state, dtype=dtype)
            reward = torch.as_tensor(reward, dtype=dtype)
            terminated = torch.as_tensor(terminated, dtype=torch.bool)
            return state, action, next_state, reward, terminated
