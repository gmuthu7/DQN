import random

import numpy as np
import torch
import torch.nn
from numpy.random import Generator


def set_random_all(seed: int) -> Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # TODO: Add cuda deterministic
    nprng = np.random.default_rng(seed)
    return nprng
