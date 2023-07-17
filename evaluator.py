from typing import Callable

import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.base_agent import Agent


class Evaluator:

    def evaluate(self, eval_env: Env, agent: Agent, num_steps: int, seed: int, callback: Callable):
        eval_env.reset(seed=seed)
        step = 0
        eval_env = RecordEpisodeStatistics(eval_env)
        state, info = eval_env.reset()
        ep_rews = []
        ep_lens = []
        while step < num_steps:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            if "episode" in info:
                last_ep_returns = info["episode"]["r"][info["_episode"]]
                last_ep_lens = info["episode"]["l"][info["_episode"]]
                ep_rews.extend(last_ep_returns)
                ep_lens.extend(last_ep_lens)
            state = next_state
            step += 1
        return callback({"eval_mean_ep_rew": np.mean(ep_rews),
                         "eval_mean_ep_len": np.mean(ep_lens)})
