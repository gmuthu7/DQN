from typing import Dict

from loggers.mlflow_logger import MlflowLogger
from loggers.ray_tune_logger import RayTuneLogger


class MlflowRayTuneLogger(MlflowLogger):

    def __init__(self, track_metric: str, tracking_uri: str, experiment_id: str, parent_run_id: str, tmp_dir: str):
        super().__init__(tracking_uri, experiment_id, parent_run_id, tmp_dir)
        self.ray_logger = RayTuneLogger(track_metric, tmp_dir)

    def log_metrics(self, params: Dict, step: int):
        super().log_metrics(params, step)
        self.ray_logger.log_metrics(params, step)
