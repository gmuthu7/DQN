from builders.config import CARTPOLE_CONFIG, SEARCH_SPACE
from scripts.run import run
from scripts.tune import ray_tune


def test_tune():
    CARTPOLE_CONFIG.update(SEARCH_SPACE)
    ray_tune(CARTPOLE_CONFIG, run)
