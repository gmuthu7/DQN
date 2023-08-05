from unittest.mock import Mock

import numpy as np
import torch


def test_get_target(dqn):
    mock = Mock()
    reward = torch.tensor([1., 1.])
    next_state = torch.rand(size=(2, 1))
    terminated = torch.tensor([True, False])
    dqn.vfa_greedy_policy = mock
    dqn.target_vfa = mock
    mock.choose_action.return_value = torch.tensor([1, 1])
    mock.val.return_value = torch.tensor([[1., 1.], [2., 3.]])
    assert torch.equal(dqn._get_target(reward, next_state, terminated, 1.),
                       torch.tensor([1., 4.]))


def test_learn(dqn):
    state = np.random.rand(2, 1)
    action = np.array([0, 1])
    next_state = np.random.rand(2, 1)
    reward = np.array([1., 2.])
    terminated = np.array([False, True])
    truncated = np.array([False, False])
    gamma = 1.
    step = 100
    dqn.learn(state, action, next_state, reward, terminated, truncated, gamma, step, lambda x: None)
