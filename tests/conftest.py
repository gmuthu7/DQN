from unittest.mock import Mock

import gymnasium as gym
import pytest
import torch

from agents.dqn import Dqn
from buffers.experience_replay import ExperienceReplay
from utility.utility import set_random_all
from vfa.neural_network import NeuralNetworkVfa


@pytest.fixture
def myseed():
    set_random_all(10)


@pytest.fixture
def env():
    num_envs = 2
    env = gym.vector.make("FrozenLake-v1", num_envs=num_envs, is_slippery=False)
    return env


@pytest.fixture
def buffer() -> ExperienceReplay:
    buffer = ExperienceReplay(2, 2)
    return buffer


@pytest.fixture()
def vfa():
    network = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.2)
    clip_val = 10
    vfa = NeuralNetworkVfa(network, loss_fn, optimizer, clip_val)
    return vfa


@pytest.fixture
def agent_mock():
    mock = Mock()
    mock.act_to_learn.side_effect = [(1, 0), (1, 0), (2, 0), (2, 0), (1, 0), (2, 0)] * 10
    mock.act.side_effect = [(1, 0), (1, 0), (2, 0), (2, 0), (1, 0), (2, 0)] * 10
    return mock


@pytest.fixture
def dqn(vfa, buffer):
    mock = Mock()
    dqn = Dqn(buffer, vfa, mock, 10, 10, 10)
    return dqn
