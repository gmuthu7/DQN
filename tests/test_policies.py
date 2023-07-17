from unittest.mock import patch, Mock

import torch

from policies.epsilon import EpsilonPolicy
from policies.greedy import GreedyPolicy


def test_greedy():
    myfn = lambda x: torch.tensor([2, 100, 4])
    policy = GreedyPolicy(myfn)
    state = torch.rand(32)
    arg = policy.choose_action(state, 0, lambda x: None)
    assert arg == 1


@patch("policies.epsilon.torch.rand", return_value=torch.tensor([0.4, 0.6, 0.7]))
def test_epsilon(somemock):
    wrapper_policy = Mock()
    wrapper_policy.choose_action.return_value = torch.tensor([0, 0, 0])
    action_sampler = lambda: torch.tensor([1, 1, 1])
    epsilon_scheduler = lambda x: 0.5
    policy = EpsilonPolicy(epsilon_scheduler, action_sampler, wrapper_policy)
    assert torch.equal(policy.choose_action(torch.tensor([21312]), 0, lambda x: None),
                       torch.tensor([1, 0, 0]))
