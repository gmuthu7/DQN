from typing import Callable

from gymnasium import Env

from agents.base_agent import Agent


class Evaluator:

    def __init__(self, eval_num_random_actions: int):
        self.eval_num_random_actions = eval_num_random_actions

    def evaluate(self, eval_env: Env, agent: Agent, num_steps: int, callback: Callable):
        step = 0
        ep_rews = []
        ep_lens = []
        while step < num_steps:
            state, info = eval_env.reset()
            ep_len = 0
            num_random = self.eval_num_random_actions
            ep_rew = 0
            while True:
                if num_random > 0:
                    num_random -= 1
                    action = 0
                else:
                    action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                ep_len += 1
                ep_rew += reward
                if terminated or truncated:
                    break
            ep_rews.append(ep_rew)
            ep_lens.append(ep_len)
            episode += 1
        return np.mean(ep_rews), {"eval_mean_ep_rew": np.mean(ep_rews),
                                  "eval_mean_ep_len": np.mean(ep_lens)}
