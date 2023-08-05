from typing import Any

from ray.air import session
from ray.air.integrations.mlflow import setup_mlflow

from loggers.mlflow_logger import MlflowLogger


class TuneMflowLogger(MlflowLogger):

    def __init__(self, log_every: int, exp_name: str, ray_str: str):
        super().__init__(log_every, exp_name)
        self.ray_str = ray_str

    def log_metric(self, key: Any, value: Any, step: int, **kwargs):
        super().log_metric(key, value, step, **kwargs)
        if key == self.ray_str:
            session.report({key: value})

    def start_run(self, **kwargs):
        setup_mlflow(
            {},
            experiment_name=self.exp_name,
            tracking_uri=self.tracking_uri,
        )

    def terminate_run(self, **kwargs):
        pass
