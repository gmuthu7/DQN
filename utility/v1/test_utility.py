import gymnasium as gym
import mlflow
import numpy as np
import pytest

from agents.base_agent import Config
from trainer import evaluate_agent
from utility.logger import MlflowLogger
from utility.utility import get_cartpole_parameters
from utility.v1.builder.dqn_trainer_builder import DqnTrainerBuilder


def test_evaluate_cartpole():
    param_dict = get_cartpole_parameters()
    logged_model = 'runs:/f2c77c9f5e3043ab8bd4a92045fd9f3e/model'
    loaded_model = mlflow.pyfunc.load_pyfunc(logged_model)
    env = gym.make(param_dict["env"], render_mode="human")
    env.reset(seed=40)
    ret = evaluate_agent(env, 5, loaded_model, num_initial_random_actions=5)
    print(ret)


if __name__ == "__main__":
    envs = gym.vector.make("FrozenLake-v1", num_envs=3, is_slippery=True)
    envs.reset()
    i = 0
    while i < 100:
        a, b, c, d, e = envs.step(np.asarray([1, 1, 1]))
        print("Step ", i, a, b, c, d, e)
        i += 1
        pass


@pytest.fixture
def driver_builder():
    return DqnTrainerBuilder()


def test_config():
    config = Config(**{"a": 1, "b": {"c": 2}})
    assert config.a == 1
    assert config.b.c == 2


def test_driver_builder_logger(driver_builder):
    driver_builder.logger(MlflowLogger, batch_size=100, log_every=100)
    print(driver_builder.l["logger"])
    assert driver_builder.o["logger"].batch_size == 100


def test_driver_builder(driver_builder):
    res = driver_builder.logger(MlflowLogger, batch_size=100, log_every=200) \
        .env(id="CartPole-v1", num_envs=4) \
        .eval(num_episodes=10, eval_freq=1000, num_initial_random_actions=10) \
        .num_steps(1e6).build()
    print(res)
