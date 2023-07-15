from abc import abstractmethod, ABC
from typing import Callable

from torch import Tensor


class Policy(ABC):
    @abstractmethod
    def choose_action(self,
                      state: Tensor,
                      step: int,
                      callback: Callable) -> Tensor:
        pass
