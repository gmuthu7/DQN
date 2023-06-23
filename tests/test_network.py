from unittest.mock import Mock

import pytest
import torch

from networks.deep_q_network import DeepQNetwork


@pytest.fixture()
def deep_q_network():
    mock = Mock()
    mock.action_space.n = 3
    mock.observation_space.shape = [10]
    return DeepQNetwork(mock)


def test_basic_q_network(deep_q_network):
    x = torch.randn((64, 10))
    logits = deep_q_network(x)
    assert logits.shape[0] == 64
    assert logits.shape[1] == deep_q_network.num_outputs
