
import torch
from torch import nn

from pbt import Member

from procedure import *
from procedure.dqn import DQNTrainingProcedure, MemoryBuffer
from procedure.hyperparams import *

from . import execute_training

from util import *

from model.dqn import Agent

from tqdm import trange

class DQNMember(Member):
    def __init__(self, env, procedure=None, mem_buff=None, use_reward_normalization=True, **args):
        procedure = procedure or DQNTrainingProcedure(env=env,
                mem_buff=mem_buff,
                use_reward_normalization=use_reward_normalization,
        )

        super().__init__(env=env, procedure=procedure, **args)

# Agent definition
def create_agent(env):
    state_size = env.observation_space.shape[-1]
    n_actions = env.action_space.n

    return Agent(state_size, n_actions).float().cuda()

def train(env_generator,
        episode_length=2048,
        max_rounds=1024,
        round_len=8,
        population_size=4,
        use_reward_normalization=True):
    """ Performs training using DQN.

    episode_length
            The number of samples to collect before optimizing
    max_rounds
            Maximum number of times to stop and attempt exploration
    round_len
            Number of episodes to execute before attempting exploration
    population_size
            Number of agents to try at any given time
    """

    generator = ConfigurationGenerator(
        # Parameters that never change
        gamma = ConstantParam(0.99),
        tau = ConstantParam(1e-3),
        
        # Loss function for optimizing the value function
        loss_fn = ConstantParam(nn.SmoothL1Loss()),
        
        # Parameters that vary discretely
        batch_size = ChoiceParam(32, 64, 128),

        # LR initially ranges between 1e-3 and 1e-5
        lr = RangeParam(3, 5, f=lambda p: 0.1 ** p),
    )

    mem_buff = MemoryBuffer()
    
    """
    # Prepoulate the memory buffer with random data
    env = env_generator()
    state = env.reset()
    for _ in trange(mem_buff.max_size, desc='Prewarming memory buffer'):
        action = random.randrange(env.action_space.n)
        next_state, reward, done = env.step(action)[:3]

        mem_buff.add(state, action, next_state, reward, done)

        if done > 0.5: env.reset()
        state = next_state
    """

    population = set([])
    for _ in range(population_size):
        env = env_generator()
        member = DQNMember(env=env,
                mem_buff=mem_buff,
                model = create_agent(env),
                hyperparams = generator.generate(),
                use_reward_normalization = use_reward_normalization,
                score = -float('inf'))

        population.add(member)

    return execute_training(population,
            max_rounds=max_rounds,
            round_len=round_len)



