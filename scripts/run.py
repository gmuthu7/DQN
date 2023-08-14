from typing import Dict

from builders.builder import Builder
from builders.cartpole_config import CARTPOLE_CONFIG
from builders.config_director import ConfigDirector


def run(config: Dict):
    config_director = ConfigDirector(config)
    builder = Builder()
    rets = config_director.direct(builder)
    trainer, device, callback, args = rets[0], rets[1], rets[2], rets[3:]
    with device:
        trainer.train(*args, callback)


if __name__ == "__main__":
    run(CARTPOLE_CONFIG)
