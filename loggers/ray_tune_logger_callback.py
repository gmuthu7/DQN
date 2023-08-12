import logging

from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback

logger = logging.getLogger(__name__)


class RayTuneLoggerCallback(LoggerCallback):

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        if failed:
            print(f"Trial {trial.logdir} failed")
            logger.warning(f"Trial {trial.logdir} failed")
