import functools
from typing import Tuple, Dict

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from ray.air import session

from agents.base_agent import Agent
from agents.buffers.experience_replay import ExperienceReplay
from agents.double_dqn import DoubleDqn
from agents.dqn import Dqn
from agents.vfa.neural_network import NeuralNetworkVfa
from configs.root_config import DEFAULT_MLFLOW_TRACKING_URI, DEFAULT_STORAGE_DIRECTORY
from loggers.mlflow_logger import MlflowLogger
from loggers.mlflow_ray_tune_logger import MlflowRayTuneLogger
from loggers.ray_tune_logger import RayTuneLogger
from policies.epsilon import EpsilonPolicy
from policies.greedy import GreedyPolicy
from simulators.evaluator import Evaluator
from simulators.trainer import Trainer
from simulators.trainer_callback import TrainerCallback


# noinspection PyAttributeOutsideInit
class Builder:

    def env(self, env_name: str, num_envs: int):
        self.train_env: VectorEnv = gym.vector.make(env_name, num_envs, wrappers=RecordEpisodeStatistics)
        self.eval_env: VectorEnv = gym.vector.make(env_name, num_envs, wrappers=RecordEpisodeStatistics)
        self.num_actions = self.train_env.single_action_space.n.item()
        self.num_obs = self.train_env.single_observation_space.shape[0]
        self.num_envs = num_envs
        return self

    def device(self, _device: str):
        torch.set_default_device(_device)

    def ray_tune_logger(self, track_metric: str):
        self.logger = RayTuneLogger(track_metric, session.get_trial_dir())
        return self

    def mlflow_ray_tune_logger(self, track_metric: str, experiment_id: str, parent_run_id: str):
        self.logger = MlflowRayTuneLogger(track_metric, DEFAULT_MLFLOW_TRACKING_URI, experiment_id, parent_run_id,
                                          session.get_trial_dir())
        return self

    def mlflow_logger(self, experiment_id: str):
        self.logger = MlflowLogger(DEFAULT_MLFLOW_TRACKING_URI, experiment_id, None,
                                   DEFAULT_STORAGE_DIRECTORY)
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
            torch.nn.Linear(num_hidden, num_hidden * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden * 2, num_hidden * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden * 4, num_hidden * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden * 2, num_hidden),
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

    def mse_loss(self):
        self.loss_fn = torch.nn.MSELoss()
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

    def annealed_epsilon(self, end_epsilon: float, anneal_finished_step: int):
        def fn(initial_epsilon: float, end_epsilon: float, anneal_finished_step: int, step: int) -> float:
            return initial_epsilon + (end_epsilon - initial_epsilon) * min(1., step / anneal_finished_step)

        if self.no_learn >= anneal_finished_step:
            raise ValueError("No learn is is > anneal_finished_step")
        initial_epsilon = (1. - end_epsilon * min(1., self.no_learn / anneal_finished_step)) / (
                1. - min(1., self.no_learn / anneal_finished_step))
        self.epsilon_scheduler = functools.partial(fn, initial_epsilon, end_epsilon, anneal_finished_step)
        return self

    def epsilon_policy(self):
        num_actions = self.num_actions
        num_envs = self.num_envs
        action_sampler = lambda: torch.randint(0, num_actions, size=(num_envs,))
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

    def trainer_callback(self, log_every: int, params: Dict):
        self.callback = TrainerCallback(self.logger, log_every, params)
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

    def build(self) -> Tuple[Trainer, TrainerCallback, VectorEnv, Agent, int, float, int]:
        return self.simulator, self.callback, self.train_env, self.agent, self._num_steps, self._gamma, self._seed
