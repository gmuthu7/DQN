import numpy as np
import torch.nn
from ray import tune

SMOKE_CONFIG = {'seed': 27, 'device': 'cpu', 'exp_name': 'DQN_Cartpole',
                'env': {'name': 'CartPole-v1', 'num_envs': 4, 'gamma': 0.99},
                'agent': {'name': 'DoubleDqn', 'initial_no_learn_steps': 100, 'update_freq': 5,
                          'target_update_freq': 6000, 'num_updates': 2,
                          'buffer': {'name': 'ExperienceReplay', 'buffer_size': 10000, 'batch_size': 1024}},
                'trainer': {'num_steps': 100000, 'eval_freq': 100, 'eval_num_episodes': 10},
                'policy': {'name': 'EpsilonPolicy',
                           'epsilon_scheduler': {'name': 'annealed_epsilon', 'end_epsilon': 0.01,
                                                 'anneal_finished_step': 200}},
                'vfa': {'name': 'NeuralNetworkVfa', 'network': {'name': 'SimpleNeuralNetwork', 'num_hidden': 64},
                        'loss_fn': {'name': 'SmoothL1Loss'},
                        'optimizer': {'name': 'RMSprop', 'lr': 0.48535532171538653}, 'clip_grad_val': 0.0},
                'logger': {'name': 'RayTuneLogger', 'log_every': 10}}

SEARCH_NUM_STEPS = 4000
NO_LEARN = 100
EVAL_FREQ = 500
SMOKE_SEARCH_SPACE = {
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
            "anneal_finished_step": tune.choice(np.arange(NO_LEARN + 100, SEARCH_NUM_STEPS, 5000))
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
