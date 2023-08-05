import copy
import os
from typing import Dict

import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from loggers.utility import set_random_all, get_auto_increment_filename, \
    ConfigFromDict

# TODO: Add variance estimate to graph
# TODO: Epsilon reduction during no-learn, lr reduction, buffer randomization

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
        "num_updates": tune.randint(1, 5)
    },
    "vfa": {
        "optimizer": {
            "lr": tune.loguniform(0.0001, 0.9),
        },
        "clip_grad_val": tune.choice([0., 5., 10.])
    },
    "buffer": {
        "name": "ExperienceReplay",
        "buffer_size": tune.choice([10000, 50000, 100000]),
        "batch_size": tune.choice([32, 256, 1024])
    },
    "policy": {
        "epsilon_scheduler": {
            "end_epsilon": tune.choice([0.1, 0.01]),
            "anneal_finished_step": 50_000
        }
    },
    "logger": {
        "log_every": 1000,
    },
    "ray": {
        "grace_period": 20,
        "max_t": 50,
        "reduction_factor": 2,
        "storage_path": os.path.expanduser("/logs"),
        ""
    }
}


def tune(search_space: Dict, _config: Dict):
    config = copy.deepcopy(_config)
    config.update(search_space)
    c = ConfigFromDict(config)
    set_random_all(c.seed)
    filename = get_auto_increment_filename(c.storage_path, c.exp_name)
    hyperopt_search = HyperOptSearch()
    hyperband_scheduler = AsyncHyperBandScheduler(grace_period=c.ray.grace_period,
                                                  max_t=c.ray.max_t, reduction_factor=c.ray.reduction_factor)
    trainable_with_resources = tune.with_resources(run, {"cpu": 0.5})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=-1,
                                    search_alg=hyperopt_search,
                                    scheduler=hyperband_scheduler,
                                    metric="score", mode="min"
                                    ),
        run_config=ray.air.RunConfig(name=c.exp_name, storage_path=filename),
    )
    results = tuner.fit()
    results.get_dataframe().plot("step", "score")