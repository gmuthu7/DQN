import copy
from typing import Dict

from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from loggers.utility import set_random_all, get_auto_increment_filename, ConfigFromDict
from scripts.config import SEARCH_SPACE, CONFIG
from scripts.run import run


def tune(search_space: Dict, _config: Dict):
    config = copy.deepcopy(_config)
    config.update(search_space)
    c = ConfigFromDict(config)
    set_random_all(c.seed)
    filename = get_auto_increment_filename(c.exp_name)
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
    )
    results = tuner.fit()
    results.get_dataframe().plot("step", "score")


if __name__ == "__main__":
    tune(SEARCH_SPACE, CONFIG)
