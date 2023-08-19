import json
import os
import time
from typing import Dict, Optional, Any

import mlflow
import cloudpickle
from matplotlib.figure import Figure
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from numpy import ndarray
from ray._private.dict import flatten_dict

from agents.base_agent import Agent
from loggers.logger import Logger


class MlflowAgentWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, a: Agent):
        self.agent = a

    def predict(self, context, model_input: ndarray, params: Optional[Dict[str, Any]] = None):
        return self.agent.act(model_input)


class MlflowLogger(Logger):

    def __init__(self, tracking_uri: str, experiment_id: str, parent_run_id: str | None, tmp_dir: str):
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        self.client = MlflowClient()
        self.tmp_dir = tmp_dir
        if parent_run_id is None:
            self.run = self.client.create_run(experiment_id, tags={"tmp_dir": tmp_dir})
        else:
            self.run = self.client.create_run(experiment_id,
                                              tags={"mlflow.parentRunId": parent_run_id, "tmp_dir": tmp_dir})
            # print("Tracking url :", f"")
        self.run_id = self.run.info.run_id

    def log_params(self, params: Dict):
        self._convert_from_numpy_to_primitive(params)
        artifact_path = "params.json"
        self.client.log_dict(self.run_id, params, artifact_path)
        params = flatten_dict(params)
        for key, value in params.items():
            self.client.log_param(self.run_id, key, value)

    def log_metrics(self, params: Dict, step: int):
        metrics = []
        for key, value in params.items():
            metrics.append(Metric(key, value, int(time.time() * 1000), step))
        metrics.append(Metric("train/step", step, int(time.time() * 1000), step))
        self.client.log_batch(run_id=self.run_id, metrics=metrics)

    def log_model(self, agent: Agent, step: int):
        mlflow.pyfunc.log_model(python_model=MlflowAgentWrapper(agent), artifact_path="model")

    def log_figure(self, fig: Figure, step: int):
        artifact_path = f"{fig._suptitle.get_text()}.png"
        self.client.log_figure(self.run_id, fig, artifact_path)

    def deinit(self):
        self.client.set_terminated(self.run_id)

    def _convert_from_numpy_to_primitive(self, d: Dict):
        for key, val in d.items():
            if isinstance(val, dict):
                self._convert_from_numpy_to_primitive(val)
            elif hasattr(val, "item"):
                d[key] = val.item()
