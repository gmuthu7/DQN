import random

import numpy as np
import torch
from numpy.random import Generator


def set_random_all(seed) -> Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    nprng = np.random.default_rng(seed)
    return nprng
