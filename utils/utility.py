import copy
import functools
from typing import Iterator

import gymnasium as gym
import torch.nn
from torch.nn import Parameter

from buffers.experience_replay import ExperienceReplay
from networks.deep_q_network import DeepQNetwork


def get_cartpole_parameters():
    return {
        "env": "CartPole-v1",
        "buffer": {
            "name": "ExperienceReplay",
            "params": {
                "buffer_size": int(1000),
                "minibatch_size": 256
            }
        },
        "q_network": "DeepQNetwork",
        "num_steps": int(3e6),
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


def construct_parameter_obj(parameters: dict):
    p = copy.deepcopy(parameters)
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
    p["env"] = gym.make(p["env"])
    p["buffer"] = buffer[p["buffer"]["name"]](**p["buffer"]["params"])
    p["q_network"] = q_network[p["q_network"]](p["env"])
    p["epsilon_scheduler"] = functools.partial(epsilon_scheduler[p["epsilon_scheduler"]["name"]],
                                               *p["epsilon_scheduler"]["params"].values())
    p["loss_fn"] = loss_fn[p["loss_fn"]]()
    p["optimizer"] = optimizer[p["optimizer"]]
    return p


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


def get_grad_norm(parameters: Iterator[Parameter]) -> float:
    grads = []
    for param in parameters:
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return torch.linalg.norm(grads).item()
