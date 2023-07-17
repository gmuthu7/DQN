from typing import Dict

import gymnasium as gym
import numpy as np

from evaluator import Evaluator


def callback1(d: Dict):
    assert d["eval_mean_ep_rew"] == 1
    assert d["eval_mean_ep_len"] == 6


def test_evaluate(env, agent_mock):
    evaluator = Evaluator()
    evaluator.evaluate(env, agent_mock, 12, 20, callback1)


def test_seeding():
    env = gym.vector.make("CartPole-v1", num_envs=2)
    env.reset(seed=10)
    state, info = env.reset()
    next_state, reward, terminated, truncated, info = env.step([1, 0])
    env.reset(seed=10)
    state2, info2 = env.reset()
    next_state2, reward2, terminated2, truncated2, info2 = env.step([1, 0])
    assert np.array_equal(state, state2)
    assert np.array_equal(next_state, next_state2)
