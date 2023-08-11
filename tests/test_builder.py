import copy

from builders.builder import Builder
from builders.config import CARTPOLE_CONFIG, SEARCH_SPACE
from builders.config_director import ConfigDirector, ConfigFromDict


def test_builder():
    builder = Builder()
    director = ConfigDirector(CARTPOLE_CONFIG)
    rets = director.direct(builder)
    return rets


def test_dict_update():
    _CONFIG = copy.deepcopy(CARTPOLE_CONFIG)
    config1 = ConfigFromDict(_CONFIG)
    assert isinstance(config1.vfa.optimizer.lr, float)
    _CONFIG.update(SEARCH_SPACE)
    config2 = ConfigFromDict(_CONFIG)
    assert config2.trainer.num_steps < config1.trainer.num_steps
