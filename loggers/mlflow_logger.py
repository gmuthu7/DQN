import os
import time
from typing import Dict

import cloudpickle
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

    def __init__(self, tracking_uri: str, experiment_id: str, parent_run_id: str, tmp_dir: str):
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        self.client = MlflowClient()
        self.tmp_dir = tmp_dir
        if parent_run_id is None:
            self.run = self.client.create_run(experiment_id)
        else:
            self.run = self.client.create_run(experiment_id, tags={"mlflow.parentRunId": parent_run_id})
        self.run_id = self.run.info.run_id

    def log_params(self, params: Dict):
        for key, value in params.items():
            self.client.log_param(self.run_id, key, value)

    def log_metrics(self, params: Dict, step: int):
        metrics = []
        for key, value in params.items():
            metrics.append(Metric(key, value, int(time.time() * 1000), step))
        metrics.append(Metric("train/step", step, int(time.time() * 1000), step))
        self.client.log_batch(run_id=self.run_id, metrics=metrics)

    def log_model(self, agent: Agent, step: int):
        pass
        # local_path = os.path.join(self.tmp_dir, "agent.pkl")
        # artifact_path = "model/agent.pkl"
        # with open(os.path.join(self.tmp_dir, local_path), "wb") as f:
        #     cloudpickle.dump(agent, f)
        # self.client.log_artifact(self.run_id, local_path, artifact_path)
        # mlflow.pyfunc.log_model(python_model=MlflowAgentWrapper(agent), artifact_path="model", **kwargs)

    def log_figure(self, fig: Figure, step: int):
        artifact_path = f"{fig._suptitle.get_text()}.png"
        self.client.log_figure(self.run_id, fig, artifact_path)
        # mlflow.log_figure(fig, fig._suptitle.get_text())

    def deinit(self):
        self.client.set_terminated(self.run_id)
