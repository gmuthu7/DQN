import os
from typing import Dict

from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from builders.config import SEARCH_SPACE, DEFAULT_STORAGE_DIRECTORY
from scripts.run import run


def tune(search_space: Dict,config:Dict):
    config.update(search_space)
    filename = get_auto_increment_filename(search_space["env"]["name"],DEFAULT_STORAGE_DIRECTORY)
    hyperopt_search = HyperOptSearch(n_initial_points=c.ray.n_initial_points)
    hyperband_scheduler = AsyncHyperBandScheduler(grace_period=c.ray.grace_period,
                                                  max_t=c.ray.max_t, reduction_factor=c.ray.reduction_factor)
    trainable_with_resources = tune.with_resources(run, {"cpu": c.ray.resource_ratio,
                                                         "gpu": c.ray.resource_ratio if c.device == "cuda" else 0})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=-1,
                                    search_alg=hyperopt_search,
                                    scheduler=hyperband_scheduler,
                                    metric="score", mode="min"),
        run_config=RunConfig(name=filename, storage_path=c.ray.storage_path),
        failure_config=
    )
    results = tuner.fit()
    results.get_dataframe().plot("step", "score")


def get_auto_increment_filename(base_filename, directory):
    file_list = [file for file in os.listdir(directory) if re.match(base_filename + r'\d+', file)]

    if not file_list:
        return base_filename + '1'

    latest_suffix = max([int(re.search(r'\d+', file).group()) for file in file_list])
    new_suffix = latest_suffix + 1

    new_filename = f"{base_filename}{new_suffix}"
    return new_filename


def get_ray_storage(exp_name: str):
    return os.path.join(DEFAULT_STORAGE_DIRECTORY, exp_name)


if __name__ == "__main__":
    tune(SEARCH_SPACE, CONFIG)
