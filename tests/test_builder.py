import copy
from unittest.mock import patch, Mock

import cloudpickle

from builders.builder import Builder
from builders.cartpole_config import CARTPOLE_CONFIG
from builders.tune_config import SEARCH_SPACE
from builders.config_director import ConfigDirector, ConfigFromDict


def test_builder():
    builder = Builder()
    director = ConfigDirector(CARTPOLE_CONFIG)
    director.direct(builder)
    return builder


@patch("builders.builder.MlflowRayTuneLogger")
def test_pickle_agent(mock: Mock):
    builder = test_builder()
    res = cloudpickle.dumps(builder.agent)


def test_dict_update():
    _CONFIG = copy.deepcopy(CARTPOLE_CONFIG)
    config1 = ConfigFromDict(_CONFIG)
    assert isinstance(config1.vfa.optimizer.lr, float)
    _CONFIG.update(SEARCH_SPACE)
    config2 = ConfigFromDict(_CONFIG)
    assert config2.trainer.num_steps < config1.trainer.num_steps
