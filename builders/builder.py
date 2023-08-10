import functools
from typing import Tuple

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv

from agents.buffers.experience_replay import ExperienceReplay
from agents.double_dqn import DoubleDqn
from agents.dqn import Dqn
from agents.vfa.neural_network import NeuralNetworkVfa
from loggers.mlflow_logger import MlflowLogger
from policies.epsilon import EpsilonPolicy
from policies.greedy import GreedyPolicy
from simulators.trainer_callback import TrainerCallback
from simulators.evaluator import Evaluator
from simulators.trainer import Trainer


# noinspection PyAttributeOutsideInit
class Builder:

    def env(self, env_name: str, num_envs: int):
        self.train_env: VectorEnv = gym.vector.make(env_name, num_envs)
        self.eval_env: VectorEnv = gym.vector.make(env_name, num_envs)
        return self

    def mlflow_logger(self, log_every: int, exp_name: str):
        self.logger = MlflowLogger(log_every, exp_name)

    def experience_replay_buffer(self, buffer_size: int, batch_size: int):
        self.buffer = ExperienceReplay(buffer_size, batch_size)

    def simple_neural_network(self, num_hidden: int):
        num_inputs = self.train_env.single_observation_space.shape[0]
        num_outputs = self.train_env.single_action_space.n
        self.network = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_outputs)
        )
        return self

    def adam_optimizer(self, **kwargs):
        self.optimizer = torch.optim.AdamW(self.network.parameters(), **kwargs)
        return self

    def rmsprop_optimizer(self, **kwargs):
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), **kwargs)
        return self

    def l1_loss(self):
        self.loss_fn = torch.nn.SmoothL1Loss()
        return self

    def neural_network_vfa(self, clip_grad_val: float):
        self.vfa = NeuralNetworkVfa(self.network, self.loss_fn, self.optimizer, None, clip_grad_val)
        return self

    def greedy_policy(self):
        self.policy = GreedyPolicy(self.vfa.val)
        return self

    def initial_no_learn_steps(self, initial_no_learn_steps: int):
        self.no_learn = initial_no_learn_steps

    def annealed_epsilon(self, end_epsilon: int, anneal_finished_step: int):
        def fn(initial_epsilon: float, end_epsilon: float, anneal_finished_step: int, step: int) -> float:
            return initial_epsilon + (end_epsilon - initial_epsilon) * min(1., step / anneal_finished_step)

        initial_epsilon = (1. - end_epsilon * min(1., self.no_learn / anneal_finished_step)) / (
                1. - min(1., self.no_learn / anneal_finished_step))
        self.epsilon_scheduler = functools.partial(fn, initial_epsilon, end_epsilon, anneal_finished_step)
        return self

    def epsilon_policy(self):
        action_sampler = lambda: torch.randint(0, self.train_env.action_space.n, size=(self.train_env.num_envs,))
        greedy_policy = GreedyPolicy(self.vfa.val)
        self.policy = EpsilonPolicy(self.epsilon_scheduler, action_sampler, greedy_policy)
        return self

    def dqn(self, update_freq: int, target_update_freq: int, num_updates: int):
        self.agent = Dqn(self.buffer, self.vfa, self.policy, update_freq, target_update_freq,
                         self.no_learn, num_updates)
        return self

    def double_dqn(self, update_freq: int, target_update_freq: int, num_updates: int):
        self.agent = DoubleDqn(self.buffer, self.vfa, self.policy, update_freq, target_update_freq,
                               self.no_learn, num_updates)
        return self

    def trainer(self, eval_freq: int, eval_num_episodes: int, seed: int):
        self.callback = TrainerCallback(self.logger)
        self.simulator = Trainer(eval_freq, eval_num_episodes, self.eval_env, Evaluator(), seed)
        return self

    def build(self) -> Tuple[VectorEnv, Dqn, Trainer, TrainerCallback]:
        return self.train_env, self.agent, self.simulator, self.callback
