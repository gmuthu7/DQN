from typing import Dict

from builders.builder import Builder
from loggers.utility import ConfigFromDict


class ConfigDirector:
    def __init__(self, config: Dict):
        self.config = config

    def direct(self, builder: Builder):
        c = ConfigFromDict(self.config)
        self._direct_logger(builder, c)
        self._direct_buffer(builder, c)
        self._direct_network(builder, c)
        self._direct_optimizer(builder, c)
        builder.l1_loss()
        builder.neural_network_vfa(c.vfa.clip_grad_val)
        builder.env(c.env.name, c.env.num_envs)
        builder.initial_no_learn_steps(c.agent.initial_no_learn_steps)
        builder.annealed_epsilon(c.policy.epsilon_scheduler.end_epsilon,
                                 c.policy.epsilon_scheduler.anneal_finished_step)
        builder.epsilon_policy()
        builder.double_dqn(c.agent.update_freq, c.agent.target_update_freq, c.agent.num_updates)
        builder.trainer(c.trainer.eval_freq, c.trainer.eval_num_episodes, c.seed)
        return builder.build()

    def _direct_optimizer(self, builder: Builder, c: ConfigFromDict):
        match c.vfa.optimizer.name:
            case "RMSprop":
                builder.rmsprop_optimizer(lr=c.vfa.optimizer.lr)
            case "Adam":
                builder.adam_optimizer(lr=c.vfa.optimizer.lr)
        return self

    def _direct_network(self, builder: Builder, c: ConfigFromDict):
        match c.vfa.network.name:
            case "SimpleNeuralNetwork":
                builder.simple_neural_network(c.vfa.network.num_hidden)
        return self

    def _direct_buffer(self, builder: Builder, c: ConfigFromDict):
        match c.agent.buffer.name:
            case "ExperienceReplay":
                builder.experience_replay_buffer(c.agent.buffer.buffer_size, c.agent.buffer.batch_size)
        return self

    def _direct_agent(self, builder: Builder, c: ConfigFromDict):
        match c.agent.name:
            case "Dqn":
                builder.dqn(c.agent.update_freq, c.agent.target_update_freq, c.agent.num_updates)
            case "DoubleDqn":
                builder.adam_optimizer(lr=c.vfa.optimizer.lr)
        return self

    def _direct_logger(self, builder: Builder, c: ConfigFromDict):
        match c.logger.name:
            case "MlflowLogger":
                builder.mlflow_logger(c.logger.log_every, c.env.exp_name)
        return self


class ConfigFromDict:
    def __init__(self, d=None):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigFromDict(value))
            else:
                setattr(self, key, value)
