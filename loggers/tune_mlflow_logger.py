from typing import Any

import mlflow
from ray.air import session
from ray.air.integrations.mlflow import setup_mlflow

from loggers.mlflow_logger import MlflowLogger


class TuneMlflowLogger(MlflowLogger):

    def __init__(self, log_every: int, ray_str: str, *args):
        super().__init__(log_every)
        self.ray_str = ray_str

    def log_metric(self, key: Any, value: Any, step: int, **kwargs):
        super().log_metric(key, value, step, **kwargs)
        if key == self.ray_str:
            session.report({key: value})

    def start_run(self, exp_name: str, run_name: str | None = None, **kwargs):
        setup_mlflow(
            {},
            experiment_name=exp_name,
            tracking_uri=self.tracking_uri,
            run_name=run_name
        )

    def terminate_run(self, **kwargs):
        mlflow.end_run()
