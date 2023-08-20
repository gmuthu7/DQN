from unittest.mock import Mock

import gymnasium as gym
import pytest
import torch

from agents.buffers.experience_replay import ExperienceReplay
from agents.double_dqn import DoubleDqn
from agents.vfa.neural_network import NeuralNetworkVfa
from configs.builder import Builder
from loggers.utility import set_random_all


@pytest.fixture
def seed():
    set_random_all(10)


@pytest.fixture
def builder():
    builder = Builder()
    builder.train_env = Mock()
    builder.train_env.single_observation_space.shape = [1]
    builder.train_env.single_action_space.n = 4
    builder.simple_neural_network(64)
    return builder


@pytest.fixture
def env():
    num_envs = 2
    env = gym.vector.make("FrozenLake-v1", num_envs=num_envs, is_slippery=False)
    return env


@pytest.fixture
def buffer() -> ExperienceReplay:
    buffer = ExperienceReplay(2, 2)
    return buffer


@pytest.fixture
def vfa(builder: Builder):
    network = builder.network
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=100)
    clip_val = 10
    vfa = NeuralNetworkVfa(network, loss_fn, optimizer, scheduler, clip_val)
    return vfa


@pytest.fixture
def agent_mock():
    mock = Mock()
    mock.act_to_learn.side_effect = [(1, 0), (1, 0), (2, 0), (2, 0), (1, 0), (2, 0)] * 10
    mock.act.side_effect = [(1, 0), (1, 0), (2, 0), (2, 0), (1, 0), (2, 0)] * 10
    return mock


@pytest.fixture
def dqn(vfa, buffer):
    policy = Mock()
    dqn = DoubleDqn(buffer, vfa, policy, 10, 10, 10, 10)
    return dqn
