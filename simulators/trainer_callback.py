from typing import Dict, Iterable, Any

import numpy as np

from agents.base_agent import Agent
from loggers.error_plotter import ErrorPlotter
from loggers.logger import Logger


class TrainerCallback:
    def __init__(self, logger: Logger, log_every: int):
        self.plotters: Dict[str, ErrorPlotter] = {}
        self.logger = logger
        self.best_eval = -1
        self.step_metrics: Dict[str, Any] = {}
        self.log_every = log_every
        self.last_logged_step = 0

    def step_start(self, step: int):
        pass

    def step_end(self, step: int, metrics: Dict):
        self.step_metrics.update(metrics)
        if (step - self.last_logged_step) >= self.log_every:
            for key, val in self.step_metrics.items():
                if isinstance(val, Iterable):
                    if key not in self.plotters:
                        self.plotters[key] = ErrorPlotter(key)
                    self.plotters[key].add_point(val, step)
                    self.logger.log_figure(self.plotters[key].plt_fig(), step)
                    self.step_metrics.pop(key)

            self.logger.log_metrics(self.step_metrics, step)
            self.step_metrics = {}

    def during_learn(self, step: int, metrics: Dict):
        self.step_metrics.update(metrics)

    def after_evaluate(self, step: int, agent: Agent, metrics: Dict):
        self.step_metrics.update(metrics)
        eval_metrics = metrics["eval_ep_rews"]
        eval_perf = np.mean(eval_metrics).item()
        if eval_perf >= self.best_eval:
            self.logger.log_model(agent)
            self.step_metrics["best_eval_ep_rew"] = eval_perf
            self.best_eval = eval_perf
