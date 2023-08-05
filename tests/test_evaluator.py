from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from simulators.evaluator import Evaluator


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
