from typing import Dict

from builders.builder import Builder
from builders.config_director import ConfigDirector
from builders.smoke_config import SMOKE_CONFIG


def run(config: Dict):
    config_director = ConfigDirector(config)
    builder = Builder()
    rets = config_director.direct(builder)
    trainer, device, callback, args = rets[0], rets[1], rets[2], rets[3:]
    with device:
        trainer.train(*args, callback)


if __name__ == "__main__":
    run(SMOKE_CONFIG)
