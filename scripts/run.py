from typing import Dict

import torch

from configs.builder import Builder
from configs.config_director import ConfigDirector


def run(config: Dict):
    print("Using device ", config["device"])
    with torch.device(config["device"]):
        config_director = ConfigDirector(config)
        builder = Builder()
        rets = config_director.direct(builder)
        trainer, callback, args = rets[0], rets[1], rets[2:]
        trainer.train(*args, callback)
