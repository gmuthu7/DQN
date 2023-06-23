import copy
from typing import Callable, Any, Tuple, Optional, Dict

import mlflow.pyfunc
import torch
import torch.nn as nn
from gymnasium import Env
from line_profiler_pycharm import profile

from buffers.experience_replay import ExperienceReplay
from utils.utility import get_grad_norm


class DqnWithExperienceReplay(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass

    def predict(self, context, model_input):
        return self.best_action(model_input)

    def __init__(self, env: Env, buffer: ExperienceReplay, q_network: nn.Module, update_freq: int,
                 target_update_freq: int, gamma: float, epsilon_scheduler: Callable[[int], float],
                 loss_fn: Callable[[Any, Any], torch.Tensor], optimizer: torch.optim.Optimizer,
                 optimizer_args: dict, **kwargs):
        self.epsilon_scheduler = epsilon_scheduler
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.q_network = q_network
        self.buffer = buffer
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.target_network = None
        self.loss_fn = loss_fn
        # noinspection PyCallingNonCallable
        self.optimizer = optimizer(self.q_network.parameters(), **optimizer_args)

    @profile
    def choose_action(self, state: Tuple, step: int) -> int:
        with torch.no_grad():
            epsilon = self.epsilon_scheduler(step)
            state = torch.tensor(state, dtype=torch.float32)
            if torch.rand(size=(1,)).item() < epsilon:
                return torch.randint(0, self.num_actions, size=(1,)).item()
            else:
                return self._greedy_action(self.q_network, state).item()

    def best_action(self, state: Tuple) -> int:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            return self._greedy_action(self.q_network, state).item()

    @profile
    def _greedy_action(self, network: nn.Module, state: torch.Tensor) -> torch.Tensor:
        return torch.argmax(network(state), dim=-1)

    def _get_target(self, breward: torch.Tensor, bnext_state: torch.Tensor, bdone: torch.Tensor) -> torch.Tensor:
        target_output = self.target_network(bnext_state)
        return breward + (1 - bdone) * self.gamma * target_output[
            torch.arange(len(target_output)), self._greedy_action(self.target_network, bnext_state)]

    @profile
    def learn(self,
              state: Tuple,
              action: int,
              next_state: Tuple,
              reward: float,
              terminated: bool,
              truncated: bool,
              step: int) -> Optional[Dict]:
        experience = self.buffer.get_experience(state, action, next_state, reward, terminated)
        self.buffer.store(experience)
        if step % self.target_update_freq == 0:
            self.target_network = copy.deepcopy(self.q_network)
        if step % self.update_freq == 0:
            with torch.no_grad():
                bstate, baction, bnext_state, breward, bdone = self.buffer.sample()
                baction, bdone, bnext_state, breward, bstate = self._get_tensors(bstate, baction, bnext_state, breward,
                                                                                 bdone)
                btarget = self._get_target(breward, bnext_state, bdone)
            bpred: torch.Tensor = self.q_network(bstate)
            bpred: torch.Tensor = bpred[torch.arange(len(bpred)), baction]
            loss = self.loss_fn(btarget, bpred)
            self.optimizer.zero_grad()
            loss.backward()
            before_norm = get_grad_norm(self.q_network.parameters())
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
            after__norm = get_grad_norm(self.q_network.parameters())
            self.optimizer.step()
            return {"train_mean_loss": loss.item(),
                    "train_mean_qfn": torch.mean(bpred).item(),
                    "train_before_clip_grad": before_norm,
                    "train_after_clip_grad": after__norm,
                    "train_epsilon": self.epsilon_scheduler(step),
                    }
        return None

    def _get_tensors(self, bstate, baction, bnext_state, breward, bdone):
        bstate = torch.as_tensor(bstate, dtype=torch.float32)
        baction = torch.as_tensor(baction).long()
        bnext_state = torch.as_tensor(bnext_state, dtype=torch.float32)
        breward = torch.as_tensor(breward, dtype=torch.float32)
        bdone = torch.as_tensor(bdone).long()
        return baction, bdone, bnext_state, breward, bstate
