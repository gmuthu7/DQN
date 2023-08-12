import os
from typing import Dict, Callable

import ray
from mlflow import MlflowClient
from ray import tune
from ray.air import RunConfig
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from builders.tune_config import SEARCH_SPACE, DEFAULT_MLFLOW_TRACKING_URI
from scripts.ray_tuner import RayTuner
from scripts.run import run


# TODO: Need a way to log checkpoint/model since its giving error now
# TODO: Disable tensorboard logging in ray tune and remove pbar/printing for performance improvement
class MlflowRayTuner(RayTuner):
    def ray_tune(self, config: Dict, train_fn: Callable) -> ResultGrid:
        # ray.init(local_mode=True)
        os.environ["MLFLOW_TRACKING_URI"] = DEFAULT_MLFLOW_TRACKING_URI
        client = MlflowClient()
        run_id = None
        try:
            storage_path = self._get_ray_storage(config["exp_name"])
            filename = self._get_auto_increment_filename("run", storage_path)
            run_id = self._setup_mlflow(client, config, filename)
            hyperopt_search = HyperOptSearch(n_initial_points=20)
            hyperband_scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", grace_period=15000,
                                                          max_t=config["trainer"]["num_steps"], reduction_factor=3)
            trainable_with_resources = tune.with_resources(train_fn, {"cpu": 0.5,
                                                                      "gpu": 1. / 32. if config[
                                                                                             "device"] == "cuda" else 0})
            tuner = tune.Tuner(
                trainable_with_resources,
                param_space=config,
                tune_config=tune.TuneConfig(num_samples=100,
                                            search_alg=hyperopt_search,
                                            scheduler=hyperband_scheduler,
                                            metric=config["logger"]["track_metric"], mode="max"),
                run_config=RunConfig(name=filename, storage_path=storage_path,
                                     verbose=0)
            )
            result_grid = tuner.fit()
            return result_grid
        finally:
            if run_id is not None:
                client.set_terminated(run_id)

    def _setup_mlflow(self, client: MlflowClient, config: Dict, run_name: str) -> str:
        experiment = client.get_experiment_by_name(config["exp_name"])
        if experiment is None:
            experiment_id = client.create_experiment(config["exp_name"])
        else:
            experiment_id = experiment.experiment_id
        r = client.create_run(experiment_id, run_name=run_name)
        run_id = r.info.run_id
        config["logger"]["parent_run_id"] = run_id
        config["logger"]["experiment_id"] = experiment_id
        return run_id


if __name__ == "__main__":
    tuner = MlflowRayTuner()
    tuner.ray_tune(SEARCH_SPACE, run)
