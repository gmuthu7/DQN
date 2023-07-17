import copy
import functools
import random

import gymnasium as gym
import numpy as np
import torch.nn
from gymnasium import Env

from buffers.experience_replay import ExperienceReplay


class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                self.__dict__.update({key: Config(**value)})
                continue
            self.__dict__.update(entries)


def set_random_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    nprng = np.random.default_rng(seed)


def get_cartpole_parameters():
    return {
        "env": {"name": "CartPole-v1", "num_envs": 4},
        "buffer": {
            "name": "ExperienceReplay",
            "params": {
                "buffer_size": int(1000),
                "minibatch_size": 256
            }
        },
        "q_network": "DeepQNetwork",
        "num_steps": int(3e6),
        "initial_no_learn_steps": 1000,
        "update_freq": 4,
        "target_update_freq": 10000,
        "eval": {
            "eval_freq": int(2.5e3),
            "num_episodes": 10
        },
        "gamma": 0.99,
        "epsilon_scheduler": {
            "name": "annealed_epsilon",
            "params": {
                "initial_epsilon": 1,
                "end_epsilon": 0.05,
                "anneal_finished_step": int(5e4)
            }
        },
        "loss_fn": "mseloss",
        "optimizer": "adam",
        "optimizer_args": {
            "lr": 1e-4
        }
    }


def annealed_epsilon(initial_epsilon: float, end_epsilon: float, anneal_finished_step: int, step: int) -> float:
    return initial_epsilon + (end_epsilon - initial_epsilon) * min(1., step / anneal_finished_step)


def construct_parameter_obj(pm: dict):
    p = copy.deepcopy(pm)
    buffer = {
        "ExperienceReplay": ExperienceReplay
    }
    epsilon_scheduler = {
        "annealed_epsilon": annealed_epsilon
    }
    q_network = {
        "DeepQNetwork": DeepQNetwork
    }
    loss_fn = {
        "mseloss": torch.nn.MSELoss
    }
    optimizer = {
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam
    }
    p["env"] = gym.vector.make(pm["env"]["name"], num_envs=pm["env"]["num_envs"])
    p["eval"]["env"]: Env = gym.make(pm["env"]["name"])
    p["buffer"] = buffer[pm["buffer"]["name"]](**pm["buffer"]["params"])
    p["q_network"] = q_network[pm["q_network"]](p["eval"]["env"].observation_space.n, p["eval"]["env"].action_space.n)
    p["epsilon_scheduler"] = functools.partial(epsilon_scheduler[pm["epsilon_scheduler"]["name"]],
                                               *pm["epsilon_scheduler"]["params"].values())
    p["loss_fn"] = loss_fn[pm["loss_fn"]]()
    p["optimizer"] = optimizer[pm["optimizer"]](p["q_network"].parameters(), **pm["optimizer_args"])
    return p


def backup_code():
    mock = Mock()
    next_state = torch.ones((num_envs, 4))
    reward = torch.ones((num_envs,))
    terminated = torch.full((num_envs,), False)
    truncated = torch.full((num_envs,), False)
    ret1 = (next_state, reward, terminated, truncated, {})


def flatten_dictionary(dictionary, parent_key='', separator='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        # Generate the new key using the parent key and separator
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten the nested dictionary
            flattened_dict.update(flatten_dictionary(value, new_key, separator))
        else:
            flattened_dict[new_key] = value

    return flattened_dict


def test_parameter_obj():
    from trainer import get_cartpole_parameters
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
    eret = evaluate_agent(None, 1, mock2)
    assert eret[1] == {'eval_mean_ep_rew': 1.0, 'eval_mean_ep_len': 1.0}
