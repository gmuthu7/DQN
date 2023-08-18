import os

import numpy as np
from ray import tune
import torch

SEARCH_NUM_STEPS = 200_000
NO_LEARN = 10_000
EVAL_FREQ = 1000
TARGET_UPDATE_MAX = 10000
# DEVICE = "mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = "cpu"
print("Using device ", DEVICE)
SEARCH_SPACE = {
    "seed": 27,
    "device": DEVICE,
    "exp_name": "DQN_Cartpole",
    "ray": {
        "max_t": SEARCH_NUM_STEPS + 50,
        "grace_period": NO_LEARN + 2 * TARGET_UPDATE_MAX + 500,
        "reduction_factor": 3,
        "num_samples": 100,
        "cpu": 1,
        "gpu": 1. / 32. if DEVICE == "cuda" else 0.
    },
    "env": {
        "name": "CartPole-v1",
        "num_envs": 32 if DEVICE == "cuda" else 4,
        "gamma": 0.99,
    },
    "agent": {
        "name": "DoubleDqn",
        "initial_no_learn_steps": NO_LEARN,
        "update_freq": tune.randint(1, 10),
        "target_update_freq": tune.choice(np.arange(500, TARGET_UPDATE_MAX, 1000)),
        "num_updates": tune.randint(1, 5),
        "buffer": {
            "name": "ExperienceReplay",
            "buffer_size": tune.randint(10_000, SEARCH_NUM_STEPS),
            "batch_size": tune.choice([32, 256, 512, 1024])
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
            "anneal_finished_step": tune.choice(np.arange(NO_LEARN + 10_000, SEARCH_NUM_STEPS - 50000, 5000))
        }
    },
    "vfa": {
        "name": "NeuralNetworkVfa",
        "network": {
            "name": "SimpleNeuralNetwork",
            "num_hidden": tune.choice([64, 128])
        },
        "loss_fn": {
            "name": "SmoothL1Loss"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": tune.loguniform(1e-5, 1e-2),
        },
        "clip_grad_val": tune.choice([0., 10., 20.])
    },
    "logger": {
        "name": "MlflowRayTuneLogger",
        "log_every": EVAL_FREQ,
        "track_metric": "eval/roll_mean_ep_ret"
    }
}
DEFAULT_STORAGE_DIRECTORY = os.path.expanduser("~/Projects/DQN/loggers/logs")
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
