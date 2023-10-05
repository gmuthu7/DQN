from scripts.run import run

CARTPOLE_CONFIG = {
    "seed": 27,
    "device": "cpu",
    "exp_name": "DQN_Cartpole",
    "env": {
        "name": "CartPole-v1",
        "num_envs": 4,
        "gamma": 0.99
    },
    "agent": {
        "name": "DoubleDqn",
        "initial_no_learn_steps": 10000,
        "update_freq": 8,
        "target_update_freq": 2000,
        "num_updates": 4,
        "buffer": {
            "name": "ExperienceReplay",
            "buffer_size": 256_000,
            "batch_size": 32
        }
    },
    "trainer": {
        "num_steps": int(1e6),
        "eval_freq": 1000,
        "eval_num_episodes": 20
    },
    "policy": {
        "name": "EpsilonPolicy",
        "epsilon_scheduler": {
            "name": "annealed_epsilon",
            "end_epsilon": 0.1,
            "anneal_finished_step": 500_000
        }
    },
    "vfa": {
        "name": "NeuralNetworkVfa",
        "network": {
            "name": "SimpleNeuralNetwork",
            "num_hidden": 128
        },
        "loss_fn": {
            "name": "MSELoss"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": 1e-5
        },
        "clip_grad_val": 50.
    },
    "logger": {
        "name": "MlflowLogger",
        "log_every": 1000,
        "track_metric": "eval/roll_10_mean_ep_ret",
        "experiment_id": "966957710883800030"
    },
    "ray": {
        "max_t": 200050,
        "grace_period": 30500,
        "reduction_factor": 3,
        "num_samples": 100,
        "cpu": 1,
        "gpu": 0
    },
}

if __name__ == "__main__":
    run(CARTPOLE_CONFIG)
