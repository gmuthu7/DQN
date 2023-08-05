import functools

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from ray import tune

from agents.dqn import Dqn
from loggers.mlflow_logger import MlflowLogger
from loggers.utility import annealed_epsilon
from policies.epsilon import EpsilonPolicy
from policies.greedy import GreedyPolicy


def test_log_model(vfa, buffer):
    def some_fn():
        return torch.randint(0, 4, size=(4,))

    env = gym.vector.make("CartPole-v1", num_envs=4)
    policy = EpsilonPolicy(
        functools.partial(annealed_epsilon, 1, 0.1, 1e4),
        some_fn,
        GreedyPolicy(vfa.val)
    )
    dqn = Dqn(buffer, vfa, policy, 10, 10, 10, 10)
    logger = MlflowLogger(100, "mlflowexamples")
    env.reset()
    logger.log_model(dqn)


def test_plot():
    x = tune.loguniform(0.001, 100).sample(size=1000)
    plt.hist(np.log10(x))
    plt.show()
    assert len(x) == 1000
