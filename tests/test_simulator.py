import copy
from typing import Dict

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import RecordEpisodeStatistics
from ray import tune

from scripts.config import SEARCH_SPACE
from scripts.run import ConfigFromDict, CONFIG
from simulators.evaluator import Evaluator
from simulators.trainer import Trainer


def test_train(env, agent_mock):
    trainer = Trainer(3, 100, agent_mock, agent_mock, 14, agent_mock)
    trainer.train(env, agent_mock, 0.99, 12, 10)
    call_args = agent_mock.learn.call_args_list[5][0]
    assert (call_args[2] == np.array([15, 0])).all()
    assert (call_args[3] == np.array([1., 0.])).all()
    assert (call_args[4] == np.array([True, False])).all()
    assert (call_args[5] == np.array([False, False])).all()
    call_args = agent_mock.log_metric.call_args_list[22][0]
    assert call_args[1] == 3.
    call_args = agent_mock.log_metric.call_args_list[23][0]
    assert call_args[1] == 0.5
    assert agent_mock.evaluate.call_count == 4


def test_agent_callback(agent_mock):
    trainer = Trainer(3, 100, agent_mock, agent_mock, 14, agent_mock)
    trainer._agent_callback(100)({"a": 1, "b": 2})
    agent_mock.log_metrics.assert_called_with({"a": 1, "b": 2}, step=100)


@pytest.mark.usefixtures("seed")
def test_evaluate_callback(agent_mock):
    evaluator = Evaluator()
    trainer = Trainer(3, 100, agent_mock, evaluator, 14, agent_mock)
    best_reward = [10]
    fn = trainer._evaluate_callback(10, agent_mock, best_reward)
    fn({"eval_ep_rew": np.full((10,), 100), "eval_ep_len": np.full((10,), 100)})
    assert best_reward[0] == 100


def test_dict():
    _CONFIG = copy.deepcopy(CONFIG)
    config1 = ConfigFromDict(_CONFIG)
    assert isinstance(config1.vfa.optimizer.lr, float)
    _CONFIG.update(SEARCH_SPACE)
    config2 = ConfigFromDict(_CONFIG)
    assert config2.trainer.num_steps < config1.trainer.num_steps


def callback1(d: Dict):
    assert np.mean(d["eval_mean_ep_rew"]) == 1
    assert np.mean(d["eval_mean_ep_len"]) == 6


def test_evalate(env, agent_mock):
    evaluator = Evaluator()
    evaluator.evaluate(env, agent_mock, 9, 20, callback1)


def test_seeding():
    env = gym.vector.make("CartPole-v1", num_envs=2)
    env = RecordEpisodeStatistics(env)
    state, info = env.reset(seed=10)
    next_state, reward, terminated, truncated, info = env.step([1, 0])
    state2, info2 = env.reset(seed=10)
    next_state2, reward2, terminated2, truncated2, info2 = env.step([1, 0])
    assert np.array_equal(state, state2)
    assert np.array_equal(next_state, next_state2)
    env2 = gym.vector.make("CartPole-v1", num_envs=2)
    env2 = RecordEpisodeStatistics(env2)
    __, _ = env2.reset(seed=17)
    __, _ = env.reset(seed=10)
    n2, *_ = env2.step([1, 0])
    __, _ = env2.reset(seed=17)
    n1, *_ = env2.step([1, 0])
    assert np.array_equal(n1, n2)
