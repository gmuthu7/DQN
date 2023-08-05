import numpy as np
import pytest
from simulators.evaluator import Evaluator
from scripts.main import ConfigFromDict, CONFIG
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
    config = ConfigFromDict(CONFIG)
    print(config.vfa.optimizer.lr)
