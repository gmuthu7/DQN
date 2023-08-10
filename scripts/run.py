from typing import Dict

import gymnasium as gym
import torch.nn

from builders.config import CONFIG
from simulators.evaluator import Evaluator
from policies.greedy import GreedyPolicy
from simulators.trainer import Trainer
from loggers.utility import ConfigFromDict, build


def run(config: Dict):
    c = ConfigFromDict(config)
    cuda_device = torch.device(c.device)
    logger = build(c.logger.name, c.logger.log_every, c.exp_name)
    try:
        with cuda_device:
            logger.start_run()
            buffer = build(c.buffer.name, c.buffer.buffer_size, c.buffer.batch_size)
            env = gym.vector.make(c.env.name, c.env.num_envs)
            eval_env = gym.vector.make(c.env.name, c.env.num_envs)
            num_actions = env.single_action_space.n
            network = build(c.vfa.network.name, env.single_observation_space.shape[0], num_actions)
            optimizer = build(c.vfa.optimizer.name, network.parameters(), lr=c.optimizer.lr)
            loss_fn = build(c.vfa.loss_fn.name)
            vfa = build(c.vfa.name, network, loss_fn, optimizer, None, c.vfa.clip_grad_val)
            epsilon_scheduler = build(c.policy.epsilon_scheduler.name, c.agent.initial_no_learn_steps,
                                      c.policy.epsilon_scheduler.end_epsilon,
                                      c.policy.epsilon_scheduler.anneal_finished_step)
            action_sampler = lambda: torch.randint(0, num_actions, size=(c.env.num_envs,))
            greedy_policy = GreedyPolicy(vfa.val)
            policy = build(c.policy.name, epsilon_scheduler, action_sampler, greedy_policy)
            agent = build(c.agent.name, buffer, vfa, policy, c.agent.update_freq, c.agent.target_update_freq,
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
