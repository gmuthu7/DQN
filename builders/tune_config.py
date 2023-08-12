import os

import numpy as np
import torch.nn
from ray import tune

SEARCH_NUM_STEPS = 217_000
NO_LEARN = 10_000
EVAL_FREQ = 500
SEARCH_SPACE = {
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
        "initial_no_learn_steps": NO_LEARN,
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
        "eval_freq": EVAL_FREQ,
        "eval_num_episodes": 10,
    },
    "policy": {
        "name": "EpsilonPolicy",
        "epsilon_scheduler": {
            "name": "annealed_epsilon",
            "end_epsilon": tune.loguniform(0.001, 0.1),
            "anneal_finished_step": tune.choice(np.arange(NO_LEARN + 10_000, SEARCH_NUM_STEPS, 5000))
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
            "lr": tune.loguniform(0.000001, 0.001),
        },
        "clip_grad_val": tune.choice([0., 10.])
    },
    "logger": {
        "name": "MlflowRayTuneLogger",
        "log_every": EVAL_FREQ,
        "track_metric": "eval/roll_mean_ep_ret"
    }
}
DEFAULT_STORAGE_DIRECTORY = os.path.expanduser("~/PycharmProjects/DQN/loggers/logs")
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
