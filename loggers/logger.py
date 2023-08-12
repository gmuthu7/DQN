from abc import abstractmethod, ABC
from typing import Dict

from matplotlib.figure import Figure

from agents.base_agent import Agent


# Facade Pattern
class Logger(ABC):

    @abstractmethod
    def log_params(self, params: Dict):
        pass

    @abstractmethod
    def log_metrics(self, params: Dict, step: int):
        pass

    @abstractmethod
    def log_figure(self, fig: Figure, step: int):
        pass

    @abstractmethod
    def log_model(self, agent: Agent, step: int):
        pass

    @abstractmethod
    def deinit(self):
        pass
