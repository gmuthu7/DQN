import os
import random
import re

import numpy as np
import torch
import torch.nn
from numpy.random import Generator

from scripts.config import DEFAULT_STORAGE_DIRECTORY


def set_random_all(seed) -> Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # TODO: Add cuda deterministic
    nprng = np.random.default_rng(seed)
    return nprng


def get_auto_increment_filename(base_filename, directory=DEFAULT_STORAGE_DIRECTORY):
    file_list = [file for file in os.listdir(directory) if re.match(base_filename + r'\d+', file)]

    if not file_list:
        return base_filename + '1'

    latest_suffix = max([int(re.search(r'\d+', file).group()) for file in file_list])
    new_suffix = latest_suffix + 1

    new_filename = f"{base_filename}{new_suffix}"
    return new_filename


def get_ray_storage(exp_name: str):
    return os.path.join(DEFAULT_STORAGE_DIRECTORY, exp_name)
