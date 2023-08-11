import functools
from typing import Tuple

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv
from ray.air import session

from agents.base_agent import Agent
from agents.buffers.experience_replay import ExperienceReplay
from agents.double_dqn import DoubleDqn
from agents.dqn import Dqn
from agents.vfa.neural_network import NeuralNetworkVfa
from loggers.tune_logger import RayTuneLogger
from policies.epsilon import EpsilonPolicy
from policies.greedy import GreedyPolicy
from simulators.evaluator import Evaluator
from simulators.trainer import Trainer
from simulators.trainer_callback import TrainerCallback


# noinspection PyAttributeOutsideInit
class Builder:

    def env(self, env_name: str, num_envs: int):
        self.train_env: VectorEnv = gym.vector.make(env_name, num_envs)
        self.eval_env: VectorEnv = gym.vector.make(env_name, num_envs)
        self.num_actions = self.train_env.single_action_space.n
        self.num_obs = self.train_env.single_observation_space.shape[0]
        self.num_envs = self.train_env.num_envs
        return self

    def device(self, _device: str):
        self._device = torch.device(_device)

    def ray_tune_logger(self):
        self.logger = RayTuneLogger(session.get_trial_dir())
        return self

    def experience_replay_buffer(self, buffer_size: int, batch_size: int):
        self.buffer = ExperienceReplay(buffer_size, batch_size)
        return self

    def simple_neural_network(self, num_hidden: int):
        num_inputs = self.num_obs
        num_outputs = self.num_actions
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
        return self

    def annealed_epsilon(self, end_epsilon: int, anneal_finished_step: int):
        def fn(initial_epsilon: float, end_epsilon: float, anneal_finished_step: int, step: int) -> float:
            return initial_epsilon + (end_epsilon - initial_epsilon) * min(1., step / anneal_finished_step)

        initial_epsilon = (1. - end_epsilon * min(1., self.no_learn / anneal_finished_step)) / (
                1. - min(1., self.no_learn / anneal_finished_step))
        self.epsilon_scheduler = functools.partial(fn, initial_epsilon, end_epsilon, anneal_finished_step)
        return self

    def epsilon_policy(self):
        action_sampler = lambda: torch.randint(0, self.num_actions, size=(self.num_envs,))
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

    def trainer_callback(self, log_every: int):
        self.callback = TrainerCallback(self.logger, log_every)
        return self

    def seed(self, val: int):
        self._seed = val
        return self

    def num_steps(self, steps: int):
        self._num_steps = steps
        return self

    def gamma(self, val: float):
        self._gamma = val
        return self

    def trainer(self, eval_freq: int, eval_num_episodes: int):
        self.simulator = Trainer(eval_freq, eval_num_episodes, self.eval_env, Evaluator(), self._seed)
        return self

    def build(self) -> Tuple[Trainer, torch.device, VectorEnv, Agent, int, float, int, TrainerCallback]:
        return self.simulator, self._device, self.train_env, self.agent, self._num_steps, self._gamma, self._seed, self.callback
