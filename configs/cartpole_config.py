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
        "target_update_freq": 10000,
        "num_updates": 4,
        "buffer": {
            "name": "ExperienceReplay",
            "buffer_size": 128_000,
            "batch_size": 1024
        }
    },
    "trainer": {
        "num_steps": int(2e6),
        "eval_freq": 1000,
        "eval_num_episodes": 10
    },
    "policy": {
        "name": "EpsilonPolicy",
        "epsilon_scheduler": {
            "name": "annealed_epsilon",
            "end_epsilon": 0.004758193119830186,
            "anneal_finished_step": 30000 * 2e6 / 200000
        }
    },
    "vfa": {
        "name": "NeuralNetworkVfa",
        "network": {
            "name": "SimpleNeuralNetwork",
            "num_hidden": 128
        },
        "loss_fn": {
            "name": "SmoothL1Loss"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": 0.000020938957328931204
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
