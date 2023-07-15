from typing import Callable, List

import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm

from agents.base_agent import Agent
from evaluator import Evaluator
from utility.logger import Logger


class Trainer:

    def __init__(self, eval_freq: int, eval_num_steps: int, eval_env: Env, evaluator: Evaluator,
                 logger: Logger):
        self.evaluator = evaluator
        self.logger = logger
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.eval_num_steps = eval_num_steps

    def train(self, env: Env, agent: Agent, gamma: float, num_steps: int):
        env = RecordEpisodeStatistics(env)
        state, info = env.reset()
        num_envs = state.shape[0]
        last_ep_returns = np.zeros(num_envs)
        last_ep_lens = np.zeros(num_envs)
        step = 0
        best_eval_reward = [-1.]
        with tqdm(total=num_steps) as pbar:
            self.logger.start_run()
            while step < num_steps:
                step += 1
                action = agent.act_to_learn(state, step, self._agent_callback(step))
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = np.where(terminated | truncated, info["final_observation"], next_state)
                agent.learn(state, action, next_state, reward, terminated, truncated, gamma,
                            step, self._agent_callback(step))
                if "episode" in info:
                    last_ep_returns = np.where(info["_episode"], info["episode"]["r"], last_ep_returns)
                    last_ep_lens = np.where(info["_episode"], info["episode"]["l"], last_ep_lens)
                self.logger.log_metric("train_ep_len", np.mean(last_ep_returns), step)  # Optimize
                self.logger.log_metric("train_ep_rew", np.mean(last_ep_lens), step)
                pbar.update()
                if step % self.eval_freq == 0:
                    self.evaluator.evaluate(self.eval_env, agent, self.eval_num_steps,
                                            self._evaluate_callback(step, agent, best_eval_reward))
                state = next_state
            self.logger.terminate_run()

    def _agent_callback(self, step: int) -> Callable:
        return lambda params: self.logger.log_metrics(step=step, **params)

    def _evaluate_callback(self, step: int, agent: Agent, best_eval_reward: List[float]) -> Callable:
        def fn(params: dict):
            self.logger.log_metrics(params, step)
            mean_ep_rew_str = "eval_mean_ep_rew"
            if mean_ep_rew_str not in params:
                raise ValueError(f"{mean_ep_rew_str} not returned by evaluator")
            if params[mean_ep_rew_str] > best_eval_reward[0]:
                self.logger.log_model(agent)
                best_eval_reward[0] = params[mean_ep_rew_str]

        return fn
