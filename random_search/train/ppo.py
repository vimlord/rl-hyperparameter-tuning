
import torch
from torch import nn

from procedure import *
from procedure.ppo import PPOTrainingProcedure
from procedure.hyperparams import *

from util import *

from model.ppo import Agent

import time

def create_agent(env):
    state_size = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    return Agent(state_size, n_actions).float().cuda()

def create_config(episode_length=2048):
    return ConfigurationGenerator(
        # Parameters that never change
        episode_length = ConstantParam(episode_length),
        n_training_steps = RangeParam(2, 4.5, f=lambda p: int(10 ** p)),

        gamma = ConstantParam(0.99),
        lmbda = ConstantParam(0.95),

        # Parameters that might be made variable
        L = ConstantParam(nn.SmoothL1Loss()),
        n_updates_per_step = ConstantParam(4),
        
        # Parameters that vary discretely
        batch_size = ChoiceParam(32, 64, 128),
        
        # Epsilon is essentially constant
        eps = RangeParam(0.1, 0.2),

        # LR initially ranges between 1e-3 and 1e-5
        lr = RangeParam(3, 5, f=lambda p: 0.1 ** p),
    ).generate()

def create_procedure(env, use_reward_normalization=True):
    return PPOTrainingProcedure(
            env=env,
            use_reward_normalization=use_reward_normalization)

def train(env_name,
        episode_length=2048,
        use_reward_normalization=True):
    env = Environment(env_name)

    agent = create_agent(env)
    config = create_config(episode_length=episode_length)
    procedure = create_procedure(env=env, use_reward_normalization=True)

    config.optimizer = create_simple_optimizer(model=agent, config=config)

    t = time.time()

    for ep in range(config.n_training_steps):
        procedure.do_single_training_round(agent, config)

        if (ep+1) % 10 == 0:
            tm = (time.time() - t) / (ep+1)
            exp = (config.n_training_steps - ep - 1) * tm
            print('Completed', ep+1, 'of', config.n_training_steps, f'({tm:.4f}s/it, remaining: {exp:.1f}s)')

    return agent

