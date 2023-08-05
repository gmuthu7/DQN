import os
import time
from typing import Any, Dict

import mlflow
from matplotlib.figure import Figure
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from numpy import ndarray

from agents.base_agent import Agent
from loggers.logger import Logger


class MlflowAgentWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, a: Agent):
        self.agent = a

    def predict(self, context, model_input: ndarray):
        return self.agent.act(model_input)


class MlflowLogger(Logger):

    def __init__(self, log_every: int, exp_name: str):
        self.tracking_uri = "http://127.0.0.1:5000"
        self.exp_name = exp_name
        self.log_every = log_every
        self.metrics = []
        self.last_step_metrics = []
        self.last_logged_step = 0
        self.last_step = 0
        self.client = MlflowClient()

    def log_metric(self, key: Any, value: Any, step: int, **kwargs):
        if step != self.last_step:
            self.last_step = step
            self.metrics.extend(self.last_step_metrics)
            self.last_step_metrics = []
            if (step - self.last_logged_step) >= self.log_every:
                self.last_logged_step = step
                self._flush_metrics()
        metric = Metric(key, value, int(time.time() * 1000), step)
        self.last_step_metrics.append(metric)

    def log_metrics(self, params: Dict, step: int, **kwargs):
        for key in params:
            self.log_metric(key, params[key], step, **kwargs)

    def _flush_metrics(self):
        self.client.log_batch(run_id=mlflow.active_run().info.run_id, metrics=self.metrics)
        self.metrics = []

    def log_model(self, agent: Agent, **kwargs):
        mlflow.pyfunc.log_model(python_model=MlflowAgentWrapper(agent), artifact_path="model", **kwargs)

    def log_params(self, params: Dict, **kwargs):
        mlflow.log_params(params)

    def log_fig(self, fig: Figure):
        mlflow.log_figure(fig, fig._suptitle.get_text())

    def start_run(self, **kwargs):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.exp_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.start_run()

    def terminate_run(self, **kwargs):
        self.metrics.extend(self.last_step_metrics)
        if len(self.metrics) > 0:
            self._flush_metrics()
        mlflow.end_run()
