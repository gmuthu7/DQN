from typing import Any, Dict

import mlflow
from matplotlib.figure import Figure
from ray.air import session, Checkpoint
from ray.tune.logger import LoggerCallback
from tensorboardX import SummaryWriter
from torch import Tensor

from agents.base_agent import Agent
from loggers.logger import Logger


class RayTuneLogger(Logger, LoggerCallback):

    def __init__(self, storage_path):
        self.checkpoint = None
        self.writer = SummaryWriter(storage_path)

    def log_params(self, params: Dict, **kwargs):
        raise NotImplementedError()

    def log_metric(self, key: Any, value, step: int, **kwargs):
        raise NotImplementedError()

    def log_metrics(self, params: Dict, step: int, **kwargs):
        p = {}
        for key, value in params.items():
            if isinstance(value, Tensor):
                p[key] = value.detach().cpu().item()
            else:
                p[key] = value
        p["training_iteration"] = step
        session.report(p, checkpoint=self.checkpoint)
        self.checkpoint = None

    def log_figure(self, fig: Figure, step: int):
        # mlflow.log_figure(fig, fig._suptitle.get_text())
        # self.writer.add_figure(fig._suptitle.get_text(), fig, step, False)
        fig.savefig(f"./{fig._suptitle.get_text()}.png")

    def log_model(self, agent: Agent, **kwargs):
        self.checkpoint = Checkpoint.from_dict({"agent": agent})
