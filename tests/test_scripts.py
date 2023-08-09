import pytest
import ray.air
from ray import tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from loggers.logger import Logger
from loggers.tune_mlflow_logger import TuneMlflowLogger
from loggers.utility import get_ray_storage, get_auto_increment_filename
from simulators.evaluator import Evaluator


def train_test_fn(logger: Logger, exp_name: str):
    def fn(config):
        try:
            logger.start_run(exp_name=exp_name)
            for i in range(500):
                score = i + config["a"] ** 2 + config["b"] ** 2
                logger.log_metric("score", score, i)
                session.report({"score": score, "step": i})
        finally:
            logger.terminate_run()

    return fn


@pytest.mark.usefixtures("seed")
def test_tune():
    def stop_fn(trial_id, d):
        return d["score"] - d["training_iteration"] < 0.5

    param_space = {
        "a": tune.uniform(0.1, 100),
        "b": tune.uniform(0.1, 100)
    }
    exp_name = "mlflowexamples"
    storage_path = get_ray_storage(exp_name)
    run_name = get_auto_increment_filename("test", storage_path)
    evaluator = Evaluator()
    logger = TuneMlflowLogger(2, evaluator.best_metric_str)
    obj_fn = train_test_fn(logger, exp_name)
    hyperopt_search = HyperOptSearch()
    hyperband_scheduler = AsyncHyperBandScheduler(brackets=1, grace_period=25, max_t=100, reduction_factor=2)
    try:
        logger.start_run(exp_name)
        trainable_with_resources = tune.with_resources(obj_fn, {"cpu": 0.5})
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=param_space,
            tune_config=tune.TuneConfig(num_samples=1,
                                        search_alg=hyperopt_search,
                                        scheduler=hyperband_scheduler,
                                        metric="score", mode="min"
                                        ),
            run_config=ray.air.RunConfig(name=run_name, storage_path=storage_path, stop=stop_fn, verbose=0),
        )
        results = tuner.fit()
        results.get_dataframe().plot("step", "score")
    finally:
        logger.terminate_run()
