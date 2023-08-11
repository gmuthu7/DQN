import os
import re
from typing import Dict, Callable
import subprocess
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from builders.config import SEARCH_SPACE, DEFAULT_STORAGE_DIRECTORY, CARTPOLE_CONFIG, DEFAULT_MLFLOW_TRACKING_URI
from scripts.run import run


# TODO: Switch best metric to rolling mean
# TODO: Nested runs
# TODO: Need a way to log figure and checkpoint/model since its giving error now
# TODO: Remove config from metrics by implementing logger callback,
# TODO: also close env so that it doesnt throw that damn error

def run_tensorboard(logdir: str):
    print(f"Running tensorboard at " + "http://localhost:6006")
    os.system("pkill -f tensorboad")
    subprocess.Popen(["tensorboard", "--logdir", f"{logdir}"])


def ray_tune(config: Dict, train_fn: Callable) -> ResultGrid:
    # ray.init(local_mode=True)
    storage_path = get_ray_storage(config["exp_name"])
    filename = get_auto_increment_filename("run", storage_path)
    hyperopt_search = HyperOptSearch(n_initial_points=20)
    hyperband_scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", grace_period=15000,
                                                  max_t=config["trainer"]["num_steps"], reduction_factor=3)
    trainable_with_resources = tune.with_resources(train_fn, {"cpu": 0.5,
                                                              "gpu": 0.5 if config["device"] == "cuda" else 0})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=100,
                                    search_alg=hyperopt_search,
                                    scheduler=hyperband_scheduler,
                                    metric="eval/roll_mean_ep_ret", mode="max"),
        run_config=RunConfig(name=filename, storage_path=storage_path,
                             callbacks=[MLflowLoggerCallback(tracking_uri=DEFAULT_MLFLOW_TRACKING_URI,
                                                             experiment_name=filename,
                                                             save_artifact=True)],
                             verbose=0),
    )
    result_grid = tuner.fit()
    return result_grid


def get_auto_increment_filename(base_filename: str, directory: str) -> str:
    file_list = [file for file in os.listdir(directory) if re.match(base_filename + r'\d+', file)]

    if not file_list:
        return base_filename + '1'

    latest_suffix = max([int(re.search(r'\d+', file).group()) for file in file_list])
    new_suffix = latest_suffix + 1

    new_filename = f"{base_filename}{new_suffix}"
    return new_filename


def get_ray_storage(exp_name: str) -> str:
    return os.path.join(DEFAULT_STORAGE_DIRECTORY, exp_name)


if __name__ == "__main__":
    CARTPOLE_CONFIG.update(SEARCH_SPACE)
    ray_tune(CARTPOLE_CONFIG, run)
