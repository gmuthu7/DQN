import functools

import torch
from torch import nn

from policies.epsilon import EpsilonPolicy
from utility.v1.builder.trainer_builder import TrainerBuilder
from vfa.neural_network import NeuralNetworkVfa


class DqnTrainerBuilder(TrainerBuilder):
    # ---------------------AGENT---------------------------------------------------------#

    def buffer(self, cls, **kwargs):
        self.l["buffer"] = self._extract_dict_from_fn(name=cls.__name__, **kwargs)
        self.a["buffer"] = cls(**kwargs)
        return self

    def target_update_freq(self, v):
        self.a["target_update_freq"] = self.l["target_update_freq"] = v
        return self

    def initial_no_learn_steps(self, v):
        self.a["initial_no_learn_steps"] = self.l["initial_no_learn_steps"] = v
        return self

    def gamma(self, v):
        self.a["gamma"] = self.l["gamma"] = v
        return self

    def neural_network_vfa(self, loss_fn, optimizer, clip_val):
        self.l["vfa"] = self._extract_dict_from_fn(name="neural_network_64", **locals())
        env = self._get_env()
        num_outputs = env.single_action_space.n
        num_inputs = env.single_observation_space.n
        network = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        self.a["vfa"] = NeuralNetworkVfa(network=network)

    def epsilon_scheduler(self, cls, **kwargs):
        self.l["epsilon_scheduler"] = self._extract_dict_from_fn(name=cls.__name__, **kwargs)
        self.a["epsilon_scheduler"] = functools.partial(cls, **kwargs)
        return self

    def egreedy_policy(self, **kwargs):
        scheduler = self.a["epsilon_scheduler"]
        del self.a["epsilon_scheduler"]
        env = self._get_env()
        sampler = lambda: torch.randint(0, env.single_action_space.n,
                                        size=(env.num_envs,), device="cpu")
        policy = EpsilonPolicy(scheduler=None, )
