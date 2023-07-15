from abc import abstractmethod, ABC
from typing import Callable, SupportsFloat

from gymnasium.core import WrapperActType
from numpy import ndarray


class Agent(ABC):
    @abstractmethod
    def act_to_learn(self, state: ndarray, step: int, callback: Callable) -> WrapperActType:
        pass

    @abstractmethod
    def act(self, state: ndarray) -> WrapperActType:
        pass

    @abstractmethod
    def learn(self,
              state: ndarray,
              action: int,
              next_state: ndarray,
              reward: SupportsFloat,
              terminated: bool,
              truncated: bool,
              gamma: float,
              step: int,
              callback: Callable):
        pass
