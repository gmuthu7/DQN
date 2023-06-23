import collections
from typing import TypeVar, Generic, Tuple

import numpy as np

Experience = collections.namedtuple("Experience", ["state", "action", "next_state", "reward", "done"])

T = TypeVar("T")


class ExperienceReplay(Generic[T]):

    def __init__(self, buffer_size: int, minibatch_size: int):
        self.minibatch_size = minibatch_size
        self.deque = collections.deque(maxlen=buffer_size)

    def store(self, experience: T):
        self.deque.append(experience)

    def sample(self) -> Tuple:
        idx = np.random.randint(0, len(self.deque), size=(self.minibatch_size,))
        t = np.asarray(self.deque, dtype=Experience)
        return np.asarray(t[idx, 0].tolist()), np.asarray(t[idx, 1].tolist()), np.asarray(
            t[idx, 2].tolist()), np.asarray(t[idx, 3].tolist()), np.asarray(t[idx, 4].tolist())

    @staticmethod
    def get_experience(*args) -> Experience:
        return Experience(*args)
