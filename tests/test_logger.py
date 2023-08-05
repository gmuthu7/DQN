import functools
import os.path
from typing import Callable

import gymnasium as gym
import numpy as np
import pytest
import ray.air
import torch
from matplotlib import pyplot as plt
from ray import tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from agents.dqn import Dqn
from loggers.error_plotter import ErrorPlotter
from loggers.mlflow_logger import MlflowLogger
from policies.epsilon import EpsilonPolicy
from policies.greedy import GreedyPolicy
from loggers.utility import annealed_epsilon
from loggers.utility import get_auto_increment_filename


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


@pytest.fixture
def obj_fn():
    def train_test_fn(config):
        for i in range(500):
            score = i + config["a"] ** 2 + config["b"] ** 2
            session.report({"score": score, "step": i})

    return train_test_fn


def test_plot():
    x = tune.loguniform(0.001, 100).sample(size=1000)
    plt.hist(np.log10(x))
    plt.show()
    assert len(x) == 1000


@pytest.mark.usefixtures("seed")
def test_tune(obj_fn: Callable):
    def stop_fn(trial_id, d):
        return d["score"] - d["training_iteration"] < 0.5

    DIRECTORY = os.path.expanduser("~/PycharmProjects/DQN/logs")
    BASE_FILENAME = "test_"
    filename = get_auto_increment_filename(DIRECTORY, BASE_FILENAME)
    param_space = {
        "a": tune.uniform(0.1, 100),
        "b": tune.uniform(0.1, 100)
    }
    hyperopt_search = HyperOptSearch()

    hyperband_scheduler = AsyncHyperBandScheduler(brackets=1, grace_period=25, max_t=100, reduction_factor=2)
    trainable_with_resources = tune.with_resources(obj_fn, {"cpu": 0.5})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=-1,
                                    search_alg=hyperopt_search,
                                    scheduler=hyperband_scheduler,
                                    metric="score", mode="min"
                                    ),
        run_config=ray.air.RunConfig(name=filename, storage_path=DIRECTORY, stop=stop_fn),
    )
    results = tuner.fit()
    results.get_dataframe().plot("step", "score")


@pytest.mark.usefixtures("seed")
def test_plotter():
    plotter = ErrorPlotter("test")
    for i in range(int(1e2)):
        plotter.add_point(np.random.randn(10000) * 1000, i)
    fig = plotter.plt_fig()
    fig.show()
    for i in range(int(1e2), int(1e4)):
        plotter.add_point(np.random.randn(10000) * 1000, i)
    fig = plotter.plt_fig()
    fig.show()
