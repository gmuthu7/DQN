from abc import abstractmethod, ABC
from typing import Any, Optional, Dict

import mlflow

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
    def log_metric(self, key: Any, value, step: Optional[int] = None, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, params: Dict, step: Optional[int] = None, **kwargs):
        pass

    @abstractmethod
    def log_model(self, agent: Agent, **kwargs):
        pass

    @abstractmethod
    def terminate_run(self, **kwargs):
        pass
