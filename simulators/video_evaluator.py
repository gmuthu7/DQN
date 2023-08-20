from typing import Callable

from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.base_agent import Agent

import numpy as np


class Evaluator:
    def evaluate(self, eval_env: Env, agent: Agent, num_episodes: int, seed: int, callback: Callable):
        eval_env = RecordEpisodeStatistics(eval_env)
        state, info = eval_env.reset(seed=seed)
        ep_rews = []
        ep_lens = []
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            if "episode" in info:
                last_ep_returns = info["episode"]["r"][info["_episode"]]
                last_ep_lens = info["episode"]["l"][info["_episode"]]
                ep_rews.extend(last_ep_returns)
                ep_lens.extend(last_ep_lens)
            if len(ep_rews) > num_episodes:
                break
            state = next_state
        return callback({"eval/ep_rets": ep_rews,
                         "eval/ep_lens": ep_lens,
                         "eval/mean_ep_ret": np.mean(ep_rews),
                         "eval/mean_ep_len": np.mean(ep_lens)})
