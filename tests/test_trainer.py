import numpy as np

from trainer import Trainer


def test_train(env, agent_mock):
    trainer = Trainer(3, 100, agent_mock, agent_mock, agent_mock)
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
