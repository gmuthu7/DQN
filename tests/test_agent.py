from unittest.mock import patch, Mock

import pytest
import torch

from agents.dqn_with_er import DqnWithExperienceReplay


@pytest.fixture()
@patch.object(DqnWithExperienceReplay, "__init__", side_effect=lambda *args: None)
def dqn_instance(mock_instance):
    dqn = DqnWithExperienceReplay()
    dqn.gamma = 1
    dqn.target_network = Mock()
    dqn.q_network = Mock()
    dqn.target_network.return_value = torch.tensor([[1, 2, 3], [1, 2, 3]])
    dqn.q_network.return_value = torch.tensor([[1, 2, 3], [1, 2, 3]])
    return dqn


def test_target(dqn_instance):
    batch_input = torch.tensor([1, 5]), torch.tensor([[1, 2], [3, 4]]), torch.tensor([0, 0])
    greedy_action = dqn_instance._greedy_action(dqn_instance.q_network, batch_input[1])
    assert torch.equal(greedy_action, torch.tensor([2, 2]))
    assert torch.equal(dqn_instance._get_target(*batch_input), torch.tensor([4, 8]))


def test_choose_and_greedy_action(dqn_instance):
    batch_input = torch.tensor([1, 5]), torch.tensor([[1, 2], [3, 4]])
    dqn_instance.q_network.return_value = torch.tensor([2, 0.3])
    with patch.object(torch, "rand") as mock:
        mock.return_value.item.return_value = 2
        with patch.object(dqn_instance, "epsilon_scheduler", return_value=1, create=True):
            action = dqn_instance.choose_action(batch_input[1][0], -1)
            assert action == 0
