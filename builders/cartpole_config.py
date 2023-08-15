CARTPOLE_CONFIG = {
    "agent": {
        "buffer": {
            "batch_size": 32,
            "buffer_size": 1e6,
            "name": "ExperienceReplay"
        },
        "initial_no_learn_steps": 10000,
        "name": "DoubleDqn",
        "num_updates": 3,
        "target_update_freq": 7500,
        "update_freq": 4
    },
    "device": "cpu",
    "env": {
        "gamma": 0.99,
        "name": "CartPole-v1",
        "num_envs": 4
    },
    "exp_name": "DQN_Cartpole",
    "logger": {
        "experiment_id": "986264420638131783",
        "log_every": 1000,
        "name": "MlflowLogger"
    },
    "policy": {
        "epsilon_scheduler": {
            "anneal_finished_step": 1e6,
            "end_epsilon": 0.028205027737306567,
            "name": "annealed_epsilon"
        },
        "name": "EpsilonPolicy"
    },
    "seed": 27,
    "trainer": {
        "eval_freq": 5000,
        "eval_num_episodes": 10,
        "num_steps": 3e6
    },
    "vfa": {
        "clip_grad_val": 0.0,
        "loss_fn": {
            "name": "SmoothL1Loss"
        },
        "name": "NeuralNetworkVfa",
        "network": {
            "name": "SimpleNeuralNetwork",
            "num_hidden": 128
        },
        "optimizer": {
            "lr": 2.688501867522739e-05,
            "name": "RMSprop"
        }
    }
}
