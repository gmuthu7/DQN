from typing import Callable, List

import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm

from agents.base_agent import Agent
from simulators.evaluator import Evaluator
from loggers.error_plotter import ErrorPlotter
from loggers.logger import Logger


class Trainer:

    def __init__(self, eval_freq: int, eval_num_episodes: int, eval_env: Env, evaluator: Evaluator,
                 eval_seed: int, logger: Logger):
        self.evaluator = evaluator
        self.logger = logger
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.eval_num_episodes = eval_num_episodes
        self.eval_seed = eval_seed
        self.eval_plotter = ErrorPlotter("eval_plot")
        self.train_plotter = ErrorPlotter("train_plot")

    def train(self, env: Env, agent: Agent, gamma: float, num_steps: int, seed: int):
        env = RecordEpisodeStatistics(env)
        state, info = env.reset(seed=seed)
        num_envs = state.shape[0]
        last_ep_returns = np.zeros(num_envs)
        last_ep_lens = np.zeros(num_envs)
        last_eval_step = 1
        step = 1
        best_eval_reward = [-1.]
        with tqdm(total=num_steps) as pbar:
            while step < num_steps:
                action = agent.act_to_learn(state, step, self._agent_callback(step))
                next_state, reward, terminated, truncated, info = env.step(action)
                idxs = terminated | truncated
                real_next_state = next_state.copy()
                if "final_observation" in info:
                    real_next_state[idxs] = np.stack(info["final_observation"][idxs])
                agent.learn(state, action, real_next_state, reward, terminated, truncated, gamma,
                            step, self._agent_callback(step))
                if "episode" in info:
                    last_ep_returns = np.where(idxs, info["episode"]["r"], last_ep_returns)
                    last_ep_lens = np.where(idxs, info["episode"]["l"], last_ep_lens)
                self.logger.log_metric("train_ep_len", np.mean(last_ep_lens), step)  # Optimize
                self.logger.log_metric("train_ep_rew", np.mean(last_ep_returns), step)
                self.train_plotter.add_point(last_ep_returns, step)
                if (step - last_eval_step) >= self.eval_freq:
                    last_eval_step = step
                    self.evaluator.evaluate(self.eval_env, agent, self.eval_num_episodes, self.eval_seed,
                                            self._evaluate_callback(step, agent, best_eval_reward))
                state = next_state
                pbar.update(num_envs)
                step += num_envs

    def _agent_callback(self, step: int) -> Callable:
        return lambda params: self.logger.log_metrics(params, step=step)

    def _evaluate_callback(self, step: int, agent: Agent, best_eval_reward: List[float]) -> Callable:
        def fn(params: dict):
            best_metrics = params[self.evaluator.best_metric_str]
            self.eval_plotter.add_point(best_metrics, step)
            self.logger.log_fig(self.eval_plotter.plt_fig())
            self.logger.log_fig(self.train_plotter.plt_fig())
            for param, val in params.items():
                self.logger.log_metric(param, np.mean(val), step)
            eval_perf = np.mean(best_metrics).item()
            if eval_perf >= best_eval_reward[0]:
                self.logger.log_model(agent)
                self.logger.log_metric("best_" + self.evaluator.best_metric_str, eval_perf, step)
                best_eval_reward[0] = eval_perf

        return fn
