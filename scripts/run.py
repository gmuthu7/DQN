from typing import Dict

import torch

from builders.builder import Builder
from builders.cartpole_config import CARTPOLE_CONFIG
from builders.config_director import ConfigDirector


def run(config: Dict):
    with torch.device(config["device"]):
        config_director = ConfigDirector(config)
        builder = Builder()
        rets = config_director.direct(builder)
        trainer, callback, args = rets[0], rets[1], rets[2:]
        trainer.train(*args, callback)


if __name__ == "__main__":
    run(CARTPOLE_CONFIG)
