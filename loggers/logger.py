from abc import abstractmethod, ABC
from typing import Any, Dict

import mlflow
from matplotlib.figure import Figure

from agents.base_agent import Agent


# Facade Pattern
class Logger(ABC):

    @abstractmethod
    def log_params(self, params: Dict, **kwargs):
        pass

    @abstractmethod
    def log_metric(self, key: Any, value, step: int, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, params: Dict, step: int, **kwargs):
        pass

    @abstractmethod
    def log_figure(self, fig: Figure, step: int):
        pass

    @abstractmethod
    def log_model(self, agent: Agent, **kwargs):
        pass
