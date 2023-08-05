import functools
import os
import random
import re

import numpy as np
import torch
import torch.nn
from numpy.random import Generator

from agents.double_dqn import DoubleDqn
from agents.dqn import Dqn
from buffers.experience_replay import ExperienceReplay
from loggers.mlflow_logger import MlflowLogger
from policies.epsilon import EpsilonPolicy
from vfa.neural_network import NeuralNetworkVfa


class ConfigFromDict:
    def __init__(self, d=None):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigFromDict(value))
            else:
                setattr(self, key, value)


def set_random_all(seed) -> Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # TODO: Add cuda deterministic
    nprng = np.random.default_rng(seed)
    return nprng


def get_auto_increment_filename(directory, base_filename):
    # Get a list of all files in the directory that match the base filename pattern
    file_list = [file for file in os.listdir(directory) if re.match(base_filename + r'\d+', file)]

    # If no matching files found, return the base filename with suffix 1
    if not file_list:
        return base_filename + '1'

    # Find the latest numeric suffix in the existing files and increment it
    latest_suffix = max([int(re.search(r'\d+', file).group()) for file in file_list])
    new_suffix = latest_suffix + 1

    # Construct the new filename with the incremented suffix
    new_filename = f"{base_filename}{new_suffix}"
    return new_filename


def annealed_epsilon(learn_step: int, end_epsilon: float, anneal_finished_step: int):
    def fn(initial_epsilon: float, end_epsilon: float, anneal_finished_step: int, step: int) -> float:
        return initial_epsilon + (end_epsilon - initial_epsilon) * min(1., step / anneal_finished_step)

    initial_epsilon = (1. - end_epsilon * min(1., learn_step / anneal_finished_step)) / (
            1. - min(1., learn_step / anneal_finished_step))
    return functools.partial(fn, initial_epsilon, end_epsilon, anneal_finished_step)


def simple_neural_network_64(num_inputs: int, num_outputs: int) -> torch.nn.Module:
    network = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_outputs)
    )
    return network


CLASSEST_LST = [MlflowLogger, ExperienceReplay, simple_neural_network_64, NeuralNetworkVfa, torch.nn.SmoothL1Loss,
                torch.optim.RMSprop, EpsilonPolicy, annealed_epsilon, DoubleDqn, Dqn]
CLASSES = {cls.__name__: cls for cls in CLASSEST_LST}


def get_class(s: str):
    return CLASSES[s]
