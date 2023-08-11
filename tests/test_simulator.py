from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from simulators.evaluator import Evaluator
from simulators.trainer import Trainer


def test_train(env, agent_mock):
    trainer = Trainer(3, 100, agent_mock, agent_mock, 14)
    trainer.train(env, agent_mock, 72, 0.99, 10, agent_mock)
    call_args = agent_mock.learn.call_args_list[5][0]
    assert (call_args[2] == np.array([15, 0])).all()
    assert (call_args[3] == np.array([1., 0.])).all()
    assert (call_args[4] == np.array([True, False])).all()
    assert (call_args[5] == np.array([False, False])).all()
    call_args = agent_mock.step_end.call_args_list[22][0]
    assert call_args[1] == 3.
    call_args = agent_mock.step_end.call_args_list[23][0]
    assert call_args[1] == 0.5
    assert agent_mock.evaluate.call_count == 4


def test_evaluate(env, agent_mock):
    def callback1(d: Dict):
        assert np.mean(d["eval_ep_rets"]) == 1
        assert np.mean(d["eval_ep_lens"]) == 6

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
