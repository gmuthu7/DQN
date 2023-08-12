from typing import Dict, Any

from ray._private.dict import flatten_dict

from agents.base_agent import Agent
from loggers.error_plotter import ErrorPlotter
from loggers.logger import Logger


class TrainerCallback:
    def __init__(self, logger: Logger, log_every: int, params: Dict):
        self.params = flatten_dict(params)
        self.plotters: Dict[str, ErrorPlotter] = {}
        self.logger = logger
        self.best_eval = -1
        self.roll_mean_ep_ret = 0.
        self.count = 0
        self.step_metrics: Dict[str, Any] = {}
        self.log_every = log_every
        self.last_logged_step = 0

    def train_start(self):
        self.logger.log_params(self.params)

    def step_start(self, step: int):
        pass

    def step_end(self, step: int, metrics: Dict):
        self.step_metrics.update(metrics)
        if (step - self.last_logged_step) >= self.log_every:
            self.last_logged_step = step
            for key, val in list(self.step_metrics.items()):
                if isinstance(val, list):
                    if key not in self.plotters:
                        self.plotters[key] = ErrorPlotter(key.replace("/", "_"))
                    self.plotters[key].add_point(val, step)
                    self.logger.log_figure(self.plotters[key].plt_fig(), step)
                    self.step_metrics.pop(key)

            self.logger.log_metrics(self.step_metrics, step)
        self.step_metrics = {}

    def during_learn(self, step: int, metrics: Dict):
        self.step_metrics.update(metrics)

    def after_evaluate(self, step: int, agent: Agent, metrics: Dict):
        self.step_metrics.update(metrics)
        eval_perf = metrics["eval/mean_ep_ret"]
        self.roll_mean_ep_ret += eval_perf
        self.count += 1
        self.step_metrics["eval/roll_mean_ep_ret"] = self.roll_mean_ep_ret / self.count
        if eval_perf >= self.best_eval:
            self.logger.log_model(agent, step)
            self.step_metrics["eval/best_ep_ret"] = eval_perf
            self.best_eval = eval_perf

    def train_end(self):
        self.logger.deinit()
