import os

import numpy as np
import torch.nn
from ray import tune

SEARCH_NUM_STEPS = 200_000
SEARCH_SPACE = {
    "agent": {
        "name": "DoubleDqn",
        "initial_no_learn_steps": 10_000,
        "update_freq": tune.randint(1, 10),
        "target_update_freq": tune.choice(np.arange(500, 10000, 1000)),
        "num_updates": tune.randint(1, 5),
        "buffer": {
            "name": "ExperienceReplay",
            "buffer_size": SEARCH_NUM_STEPS,
            "batch_size": tune.choice([32, 256, 1024])
        },
    },
    "trainer": {
        "num_steps": SEARCH_NUM_STEPS,
        "eval_freq": 500,
        "eval_num_episodes": 10,
    },
    "policy": {
        "name": "EpsilonPolicy",
        "epsilon_scheduler": {
            "name": "annealed_epsilon",
            "end_epsilon": tune.choice([0.1, 0.01]),
            "anneal_finished_step": tune.choice(np.arange(10_000, SEARCH_NUM_STEPS, 5000))
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
            "lr": tune.loguniform(0.00001, 0.001),
        },
        "clip_grad_val": tune.choice([0., 5., 10.])
    },
    "logger": {
        "name": "RayTuneLogger",
        "log_every": 500
    }
}

# TODO: lr reduction, buffer randomization
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
        "initial_no_learn_steps": 200,
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
        "name": "RayTuneLogger",
        "log_every": 10
    }
}
DEFAULT_STORAGE_DIRECTORY = os.path.expanduser("~/PycharmProjects/DQN/loggers/logs")
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
