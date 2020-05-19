
import torch
from torch import nn

from procedure import *
from procedure.dqn import DQNTrainingProcedure, MemoryBuffer
from procedure.hyperparams import *

from util import *

from model.dqn import Agent

import time

def create_agent(env):
    state_size = env.observation_space.shape[-1]
    n_actions = env.action_space.n
    
    agent = Agent(state_size, n_actions).float()
    return agent.cuda() if HAS_CUDA else agent

def create_config():
    return ConfigurationGenerator(
        # Parameters that never change
        n_training_steps = RangeParam(2, 4.5, f=lambda p: int(10 ** p)),

        gamma = ConstantParam(0.99),
        tau = ConstantParam(1e-3),
        
        # Loss function for optimizing the value function
        loss_fn = ConstantParam(nn.SmoothL1Loss()),
        
        # Parameters that vary discretely
        batch_size = ChoiceParam(32, 64, 128),

        # LR initially ranges between 1e-3 and 1e-5
        lr = RangeParam(3, 5, f=lambda p: 0.1 ** p),
    ).generate()

def create_procedure(env, use_reward_normalization=True):
    mem_buff = MemoryBuffer()

    return DQNTrainingProcedure(
            env=env,
            mem_buff=mem_buff,
            use_reward_normalization=use_reward_normalization)

def train(env_name,
        episode_length=2048,
        use_reward_normalization=True):
    env = Environment(env_name)

    agent = create_agent(env)
    config = create_config()
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

