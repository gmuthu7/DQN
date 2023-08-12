from typing import Dict

from matplotlib.figure import Figure
from ray.air import session
from tensorboardX import SummaryWriter
from torch import Tensor

from agents.base_agent import Agent
from loggers.logger import Logger


class RayTuneLogger(Logger):

    def __init__(self, track_metric_str: str, storage_path):
        self.checkpoint = None
        self.writer = SummaryWriter(storage_path)
        self.track_metric_str = track_metric_str
        self.last_track_metric_val = 0.

    def log_params(self, params: Dict):
        pass

    def log_metrics(self, params: Dict, step: int):
        p = {}
        if self.track_metric_str not in params:
            p[self.track_metric_str] = self.last_track_metric_val
        self.last_track_metric_val = p[self.track_metric_str]
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

    def log_model(self, agent: Agent, step: int):
        # self.checkpoint = Checkpoint.from_dict({"agent": agent})
        pass

    def deinit(self):
        pass
