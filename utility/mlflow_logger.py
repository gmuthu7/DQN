import os
import time
from typing import Any, Optional, Dict

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from numpy import ndarray

from agents.base_agent import Agent
from utility.logger import Logger


class MlflowLogger(Logger):

    def __init__(self, batch_size: int, log_every: int, exp_name: str):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = exp_name
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        self.log_every = log_every
        self.batch_size = batch_size
        self.metrics = []
        self.client = MlflowClient()

    def _log_metric(self, **kwargs):
        self.client.log_batch(run_id=mlflow.active_run().info.run_id, metrics=self.metrics)
        self.metrics = []

    def log_metric(self, key: Any, value: Any, step: Optional[int] = 9, **kwargs):
        if step % self.log_every == 0:
            metric = Metric(key, value, int(time.time() * 1000), step)
            self.metrics.append(metric)
            if len(self.metrics) >= self.batch_size:
                self._log_metric(**kwargs)

    def log_metrics(self, params: Dict, step: Optional[int] = 0, **kwargs):
        for key in params:
            self.log_metric(key, params[key], step, **kwargs)

    def log_model(self, agent: Agent, **kwargs):
        class MlflowAgentWrapper(mlflow.pyfunc.PythonModel):

            def __init__(self, agent: Agent):
                self.agent = agent

            def predict(self, context, model_input: ndarray):
                return agent.act(model_input)

        mlflow.pyfunc.log_model(python_model=MlflowAgentWrapper(agent), artifact_path="model", **kwargs)

    def log_params(self, params: Dict, **kwargs):
        mlflow.log_params(params)

    def start_run(self, **kwargs):
        mlflow.start_run()

    def terminate_run(self, **kwargs):
        if len(self.metrics) > 0:
            self._log_metric(**kwargs)
        mlflow.end_run()
