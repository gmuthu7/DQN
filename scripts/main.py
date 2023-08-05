from typing import Dict

import gymnasium as gym
import torch.nn

from scripts.config import CONFIG
from simulators.evaluator import Evaluator
from policies.greedy import GreedyPolicy
from simulators.trainer import Trainer
from loggers.utility import get_class, ConfigFromDict


def run(config: Dict):
    c = ConfigFromDict(config)
    logger_c = get_class(c.logger.name)
    buffer_c = get_class(c.buffer.name)
    network_c = get_class(c.vfa.network.name)
    optimizer_c = get_class(c.vfa.optimizer.name)
    loss_fn_c = get_class(c.vfa.loss_fn.name)
    policy_c = get_class(c.policy.name)
    epsilon_scheduler_c = get_class(c.policy.epsilon_scheduler.name)
    agent_c = get_class(c.agent.name)
    vfa_c = get_class(c.vfa.name)

    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logger_c(c.logger.log_every, c.exp_name)
    try:
        with cuda_device:
            logger.start_run()
            buffer = buffer_c(c.buffer.buffer_size, c.buffer.batch_size)
            env = gym.vector.make(c.env.name, c.env.num_envs)
            eval_env = gym.vector.make(c.env.name, c.env.num_envs)
            num_actions = env.single_action_space.n
            network = network_c(env.single_observation_space.shape[0], num_actions)
            optimizer = optimizer_c(network.parameters(), lr=c.optimizer.lr)
            loss_fn = loss_fn_c()
            vfa = vfa_c(network, loss_fn, optimizer, None, c.vfa.clip_grad_val)
            epsilon_scheduler = epsilon_scheduler_c(c.agent.initial_no_learn_steps,
                                                    c.policy.epsilon_scheduler.end_epsilon,
                                                    c.policy.epsilon_scheduler.anneal_finished_step)
            action_sampler = lambda: torch.randint(0, num_actions, size=(c.env.num_envs,))
            greedy_policy = GreedyPolicy(vfa.val)
            policy = policy_c(epsilon_scheduler, action_sampler, greedy_policy)
            agent = agent_c(buffer, vfa, policy, c.agent.update_freq, c.agent.target_update_freq,
                            c.agent.initial_no_learn_steps,
                            c.agent.num_updates)
            trainer = Trainer(c.trainer.eval_freq, c.trainer.eval_num_episodes, eval_env, Evaluator(), c.seed,
                              logger)
            logger.log_params(c)
            trainer.train(env, agent, c.env.gamma, c.trainer.num_steps, c.seed)
    finally:
        logger.terminate_run()


if __name__ == "__main__":
    run(CONFIG)
