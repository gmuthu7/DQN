from typing import Dict

import torch.cuda

from builders.builder import Builder
from builders.config import CARTPOLE_CONFIG
from builders.config_director import ConfigDirector
from builders.smoke_config import SMOKE_CONFIG


def run(config: Dict):
    config_director = ConfigDirector(config)
    builder = Builder()
    rets = config_director.direct(builder)
    trainer, device, args = rets[0], rets[1], rets[2:]
    with device:
        trainer.train(*args)


if __name__ == "__main__":
    run(SMOKE_CONFIG)
