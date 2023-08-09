from builders.dqn_builder import DqnBuilder
from loggers.utility import ConfigFromDict


class ConfigDirector:
    def __init__(self, builder: DqnBuilder):
        self.builder = builder

    def direct_from_config(self, config: dict):
        c = ConfigFromDict(config)
        self._direct_logger(c)
        self._direct_buffer(c)
        self._direct_network(c)
        self._direct_optimizer(c)
        self.builder.l1_loss()
        self.builder.neural_network_vfa(c.vfa.clip_grad_val)

    def _direct_optimizer(self, c: ConfigFromDict):
        match c.vfa.optimizer.name:
            case "RMSprop":
                self.builder.rmsprop_optimizer(lr=c.vfa.optimizer.lr)
            case "Adam":
                self.builder.adam_optimizer(lr=c.vfa.optimizer.lr)
        return self

    def _direct_network(self, c: ConfigFromDict):
        match c.vfa.network.name:
            case "SimpleNeuralNetwork":
                self.builder.simple_neural_network(c.vfa.network.num_hidden)
        return self

    def _direct_buffer(self, c: ConfigFromDict):
        match c.agent.buffer.name:
            case "ExperienceReplay":
                self.builder.experience_replay_buffer(c.agent.buffer.buffer_size, c.agent.buffer.batch_size)
        return self

    def _direct_agent(self, c: ConfigFromDict):
        match c.agent.name:
            case "Dqn":
                self.builder.dqn(c.agent.update_freq, c.agent.target_update_freq, c.agent.num_updates)
            case "DoubleDqn":
                self.builder.adam_optimizer(lr=c.vfa.optimizer.lr)
        return self

    def _direct_logger(self, c: ConfigFromDict):
        match c.logger.name:
            case "MlflowLogger":
                self.builder.mlflow_logger(c.logger.log_every, c.env.exp_name)
        return self


class ConfigFromDict:
    def __init__(self, d=None):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigFromDict(value))
            else:
                setattr(self, key, value)
