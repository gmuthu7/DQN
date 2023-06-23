import time
from abc import abstractmethod
from typing import Any, Optional, Dict

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


class Logger:

    @abstractmethod
    def log_params(self, params: Dict, **kwargs):
        pass

    @abstractmethod
    def log_metric(self, key: Any, value, step: Optional[int] = None, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, params: Dict, step: Optional[int] = None, **kwargs):
        pass

    @abstractmethod
    def log_model(self, model: Any, **kwargs):
        pass

    @abstractmethod
    def flush(self, **kwargs):
        pass


class MLFlowLogger(Logger):

    def __init__(self, batch_size: int, log_every: int):
        self.log_every = log_every
        self.batch_size = batch_size
        self.metrics = []
        self.client = MlflowClient()

    def _log_metric(self, **kwargs):
        self.client.log_batch(run_id=mlflow.active_run().info.run_id, metrics=self.metrics)
        self.metrics = []

    def log_metric(self, key: Any, value: Any, step: Optional[int] = 9, **kwargs):
        metric = Metric(key, value, int(time.time() * 1000), step)
        self.metrics.append(metric)
        if len(self.metrics) >= self.batch_size:
            self._log_metric(**kwargs)

    def log_metrics(self, params: Dict, step: Optional[int] = 0, **kwargs):
        for key in params:
            self.log_metric(key, params[key], step, **kwargs)

    def log_model(self, model: Any, **kwargs):
        mlflow.pyfunc.log_model(python_model=model, artifact_path="model", **kwargs)

    def log_params(self, params: Dict, **kwargs):
        mlflow.log_params(params)

    def flush(self, **kwargs):
        if len(self.metrics) > 0:
            self._log_metric(**kwargs)
