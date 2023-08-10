import functools

import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm

from agents.base_agent import Agent
from simulators.trainer_callback import TrainerCallback
from simulators.evaluator import Evaluator


class Trainer:

    def __init__(self, eval_freq: int, eval_num_episodes: int, eval_env: Env, evaluator: Evaluator,
                 eval_seed: int):
        self.evaluator = evaluator
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.eval_num_episodes = eval_num_episodes
        self.eval_seed = eval_seed

    def train(self, env: Env, agent: Agent, gamma: float, num_steps: int, seed: int, callback: TrainerCallback):
        env = RecordEpisodeStatistics(env)
        state, info = env.reset(seed=seed)
        num_envs = state.shape[0]
        last_ep_returns = np.zeros(num_envs)
        last_ep_lens = np.zeros(num_envs)
        last_eval_step = 1
        step = 1
        with tqdm(total=num_steps) as pbar:
            while step < num_steps:
                callback.step_start(step)
                action = agent.act_to_learn(state, step, functools.partial(callback.during_learn, step))
                next_state, reward, terminated, truncated, info = env.step(action)
                idxs = terminated | truncated
                real_next_state = next_state.copy()
                if "final_observation" in info:
                    real_next_state[idxs] = np.stack(info["final_observation"][idxs])
                if "episode" in info:
                    last_ep_returns = np.where(idxs, info["episode"]["r"], last_ep_returns)
                    last_ep_lens = np.where(idxs, info["episode"]["l"], last_ep_lens)

                agent.learn(state, action, real_next_state, reward, terminated, truncated, gamma,
                            step, functools.partial(callback.during_learn, step))

                if (step - last_eval_step) >= self.eval_freq:
                    last_eval_step = step
                    self.evaluator.evaluate(self.eval_env, agent, self.eval_num_episodes, self.eval_seed,
                                            functools.partial(callback.after_evaluate, step, agent))
                state = next_state
                pbar.update(num_envs)
                callback.step_end(step, {"train_ep_lens": last_ep_lens, "train_ep_rets": last_ep_returns})
                step += num_envs
