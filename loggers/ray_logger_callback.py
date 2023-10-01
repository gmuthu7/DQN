import logging

from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback

logger = logging.getLogger(__name__)


class RayTuneLoggerCallback(LoggerCallback):
    def __init__(self):
        self.failed_trials = []

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        if failed:
            logger.error(f"Trial {trial.logdir} failed")
            self.failed_trials.append(trial.logdir)

    def print_results(self):
        logger.info("Failed Trials ", len(self.failed_trials))
        for trial in self.failed_trials:
            print(trial.logdir)
