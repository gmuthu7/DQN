import torch
import torch.nn as nn
from gymnasium import Env


class DeepQNetworkForImage(nn.Module):
    def __init__(self, env: Env):
        super().__init__()
        self.num_outputs = env.action_space.n
        self.num_inputs = env.observation_space.n

    def forward(self, x):
        pass


class DeepQNetwork(nn.Module):

    def __init__(self, env: Env):
        super().__init__()
        self.num_outputs = env.action_space.n
        self.num_inputs = env.observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(self.num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return logits
