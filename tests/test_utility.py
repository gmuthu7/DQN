from unittest.mock import patch, Mock

import gymnasium as gym

from agents.dqn_with_er import DqnWithExperienceReplay
from main import evaluate_agent
from utils.utility import construct_parameter_obj


def test_parameter_obj():
    from main import get_cartpole_parameters
    parameters = get_cartpole_parameters()
    p = construct_parameter_obj(parameters)
    DqnWithExperienceReplay(**p)
    p["epsilon_scheduler"](0)


@patch.object(gym, "make")
def test_evaluate_agent(mock):
    mock1 = Mock()
    mock.return_value = mock1
    mock1.reset.return_value = 1, 2
    mock1.step.side_effect = [(1, 1, True, True, ""), (1, 1, True, True, "")]
    mock2 = Mock()
    mock2.best_action.return_value = 0
    eret = evaluate_agent("test", mock2, 1)
    assert eret[1] == {'eval_mean_ep_rew': 1.0, 'eval_mean_ep_len': 1.0}
