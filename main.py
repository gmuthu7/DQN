import os
from typing import Optional

import gymnasium as gym
import mlflow
import numpy as np
import torch
from line_profiler_pycharm import profile
from tqdm import tqdm

from agents.double_dqn_with_er import DoubleDqnWithExperienceReplay
from agents.dqn_with_er import DqnWithExperienceReplay
from utils.logger import MLFlowLogger
from utils.utility import construct_parameter_obj, flatten_dictionary, get_cartpole_parameters

os.environ["MLFLOW_EXPERIMENT_NAME"] = "DQN"
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# %%

def evaluate_agent(env_name: str, num_episodes: int, agent: Optional[DqnWithExperienceReplay] = None):
    episode = 0
    env = gym.make(env_name)
    ep_rews = []
    ep_lens = []
    while episode < num_episodes:
        state, _ = env.reset()
        ep_len = 0
        ep_rew = 0
        while True:
            action = agent.best_action(state)
            next_state, reward, terminated, truncated, *_ = env.step(action)
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


# %%
@profile
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = MLFlowLogger(10000, 1000)
    with torch.device(device):
        with mlflow.start_run():
            param_dict = get_cartpole_parameters()
            logger.log_params(flatten_dictionary(param_dict))
            parameters = construct_parameter_obj(param_dict)
            agent = DoubleDqnWithExperienceReplay(**parameters)
            env = parameters["env"]
            eval_freq = parameters["eval"]["eval_freq"]
            num_steps = parameters["num_steps"]
            step = 0
            best_eval_reward = -1.
            with tqdm(total=num_steps) as pbar:
                while step < num_steps:
                    state, _ = env.reset()
                    ep_len = 0
                    ep_rew = 0
                    while step < num_steps:
                        action = agent.choose_action(state, step)
                        next_state, reward, terminated, truncated, *_ = env.step(action)
                        ret = agent.learn(state, action, next_state, reward, terminated, truncated,
                                          step)
                        ep_len += 1
                        ep_rew += reward
                        step += 1
                        pbar.update(1)
                        if ret:
                            logger.log_metrics(ret, step)
                        if step % eval_freq == 0:
                            erew, eret = evaluate_agent(param_dict["env"], agent, parameters["eval"]["num_episodes"])
                            if erew >= best_eval_reward:
                                logger.log_model(agent)
                                best_eval_reward = erew
                            logger.log_metrics(eret, step)
                        if terminated or truncated:
                            logger.log_metric("train_ep_len", ep_len, step)
                            logger.log_metric("train_ep_rew", ep_rew, step)
                            break
                        state = next_state


if __name__ == "__main__":
    run()
