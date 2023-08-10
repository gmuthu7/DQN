from typing import Any, Dict

import numpy as np
from matplotlib.figure import Figure
from ray.air import session, Checkpoint
from tensorboardX import SummaryWriter

from agents.base_agent import Agent
from loggers.logger import Logger


class RayTuneLogger(Logger):

    def __init__(self, storage_path: str):
        self.writer = SummaryWriter(storage_path)
        self.checkpoint = None

    def log_params(self, params: Dict, **kwargs):
        raise NotImplementedError()

    def log_metric(self, key: Any, value, step: int, **kwargs):
        raise NotImplementedError()

    def log_metrics(self, params: Dict, step: int, **kwargs):
        session.report(params, checkpoint=self.checkpoint)
        self.checkpoint = None

    def log_figure(self, fig: Figure, step: int):
        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = img / 255.0
        img = np.swapaxes(img, 0, 2)

        self.writer.add_image(fig._suptitle.get_text(), img, step)

    def log_model(self, agent: Agent, **kwargs):
        self.checkpoint = Checkpoint.from_dict({"agent": agent})
