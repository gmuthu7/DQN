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
    def start_run(self, **kwargs):
        mlflow.start_run()

    @abstractmethod
    def log_metric(self, key: Any, value, step: int, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, params: Dict, step: int, **kwargs):
        pass

    @abstractmethod
    def log_fig(self, fig: Figure):
        pass

    @abstractmethod
    def log_model(self, agent: Agent, **kwargs):
        pass

    @abstractmethod
    def terminate_run(self, **kwargs):
        pass
