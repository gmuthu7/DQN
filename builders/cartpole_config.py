import torch

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
