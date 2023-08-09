import os

import numpy as np
import torch.nn
from ray import tune

# TODO: lr reduction, buffer randomization

SEARCH_SPACE = {
    "trainer": {
        "num_steps": 100_000,
        "eval_freq": 5000,
        "eval_num_episodes": 10,
    },
    "agent": {
        "initial_no_learn_steps": tune.choice(np.arange(10000, 50000, 10000)),
        "update_freq": tune.randint(1, 10),
        "target_update_freq": tune.choice(np.arange(1000, 10000, 1000)),
        "num_updates": tune.randint(1, 5),
        "buffer": {
            "buffer_size": tune.choice([10000, 50000, 100000]),
            "batch_size": tune.choice([32, 256, 1024])
        },
    },
    "vfa": {
        "optimizer": {
            "lr": tune.loguniform(0.0001, 0.9),
        },
        "clip_grad_val": tune.choice([0., 5., 10.])
    },
    "policy": {
        "epsilon_scheduler": {
            "end_epsilon": tune.choice([0.1, 0.01]),
            "anneal_finished_step": 50_000
        }
    },
    "logger": {
        "name": "TuneMflowLogger",
        "log_every": 1000
    },
    "ray": {
        "grace_period": 20,
        "max_t": 50,
        "reduction_factor": 2,
        "resource_ratio": 0.5,
        "num_samples": 100,
        "n_initial_points": 10
    }
}
CARTPOLE_CONFIG = {
    "seed": 27,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "exp_name": "DQN_Cartpole",
    "env": {
        "name": "CartPole-v1",
        "num_envs": 4,
        "gamma": 0.99,
    },
    "agent": {
        "name": "DoubleDqn",
        "initial_no_learn_steps": 50_000,
        "update_freq": 4,
        "target_update_freq": 10_000,
        "num_updates": 1,
        "buffer": {
            "name": "ExperienceReplay",
            "buffer_size": 1_000_000,
            "batch_size": 32
        },
    },
    "trainer": {
        "num_steps": 3_000_000,
        "eval_freq": 5000,
        "eval_num_episodes": 16,
    },
    "policy": {
        "name": "EpsilonPolicy",
        "epsilon_scheduler": {
            "name": "annealed_epsilon",
            "end_epsilon": 0.1,
            "anneal_finished_step": 1_000_000
        }
    },
    "vfa": {
        "name": "NeuralNetworkVfa",
        "network": {
            "name": "SimpleNeuralNetwork",
            "num_hidden": 64
        },
        "loss_fn": {
            "name": "SmoothL1Loss"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": 0.0025,
        },
        "clip_grad_val": 0.
    },
    "logger": {
        "name": "MflowLogger",
        "log_every": 5000
    }
}

DEFAULT_STORAGE_DIRECTORY = os.path.expanduser("~/PycharmProjects/DQN/loggers/logs")
