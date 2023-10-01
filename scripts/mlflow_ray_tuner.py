import os
from typing import Dict, Callable, Tuple

import ray
from mlflow import MlflowClient
from ray import tune
from ray.air import RunConfig
from ray.tune import ResultGrid
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from configs.root_config import DEFAULT_MLFLOW_TRACKING_URI
from loggers.ray_logger_callback import RayTuneLoggerCallback
from scripts.ray_tuner import RayTuner


class MlflowRayTuner(RayTuner):
    def ray_tune(self, config: Dict, train_fn: Callable) -> ResultGrid:
        ray.init(local_mode=True)
        run_id = None
        client = None
        try:
            storage_path = self._get_ray_storage(config["exp_name"])
            filename = self._get_auto_increment_filename("run", storage_path)
            run_id, client = self._setup_mlflow(config, filename)
            hyperopt_search = HyperOptSearch(n_initial_points=20)
            hyperband_scheduler = AsyncHyperBandScheduler(time_attr="training_iteration",
                                                          grace_period=config["ray"]["grace_period"],
                                                          max_t=config["ray"]["max_t"] + 50,
                                                          reduction_factor=config["ray"]["reduction_factor"])
            trainable_with_resources = tune.with_resources(train_fn, {"cpu": config["ray"]["cpu"],
                                                                      "gpu": config["ray"]["gpu"]})
            tuner = tune.Tuner(
                trainable_with_resources,
                param_space=config,
                tune_config=tune.TuneConfig(num_samples=config["ray"]["num_samples"],
                                            search_alg=hyperopt_search,
                                            scheduler=hyperband_scheduler,
                                            metric=config["logger"]["track_metric"], mode="max"),
                run_config=RunConfig(name=filename, storage_path=storage_path,
                                     callbacks=[RayTuneLoggerCallback()],
                                     verbose=0)
            )
            result_grid = tuner.fit()
            return result_grid
        finally:
            if run_id is not None:
                client.set_terminated(run_id)

    def _setup_mlflow(self, config: Dict, run_name: str) -> Tuple[str, MlflowClient]:
        os.environ["MLFLOW_TRACKING_URI"] = DEFAULT_MLFLOW_TRACKING_URI
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
        client = MlflowClient()
        experiment = client.get_experiment_by_name(config["exp_name"])
        if experiment is None:
            experiment_id = client.create_experiment(config["exp_name"])
        else:
            experiment_id = experiment.experiment_id
        r = client.create_run(experiment_id, run_name=run_name)
        run_id = r.info.run_id
        config["logger"]["parent_run_id"] = run_id
        config["logger"]["experiment_id"] = experiment_id
        return run_id, client
