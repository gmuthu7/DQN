from typing import Dict

import gymnasium
from utility.base_model import BaseModel


class TrainerBuilder(BaseModel):
    o: Dict = {}
    l: Dict = {}
    a: Dict = {}

    def num_steps(self, v):
        self.o["num_steps"] = self.l["num_steps"] = v
        return self

    def logger(self, cls, **kwargs):
        self.l["logger"] = self._extract_dict_from_fn(name=cls.__name__, **kwargs)
        self.o["logger"] = cls(**kwargs)
        return self

    def env(self, **kwargs):
        self.l["env"] = kwargs
        self.o["env"] = gymnasium.make_vec(**kwargs)
        return self

    def eval(self, **kwargs):
        self.l["eval"] = self.o["eval"] = kwargs
        self.o["eval"]["env"] = gymnasium.make_vec(**self.l["env"])
        return self

    def _extract_dict_from_fn(self, **kwargs):
        return kwargs

    def _get_env(self):
        if "env" not in self.o:
            raise ValueError("Buid env before building policy")
        if not isinstance(self.o["env"].single_action_space, Discrete):
            raise ValueError("Not discrete action space")
        return self.o["env"]

    def build(self):
        return self.l
