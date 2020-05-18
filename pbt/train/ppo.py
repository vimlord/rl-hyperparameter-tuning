
import torch
from torch import nn

from pbt import Member

from procedure import *
from procedure.ppo import PPOTrainingProcedure
from procedure.hyperparams import *

from . import execute_training

from util import *

from model.ppo import Agent

class PPOMember(Member):
    def __init__(self, env, procedure=None, use_reward_normalization=True, **args):
        procedure = procedure or PPOTrainingProcedure(env,
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
    """ Performs training using PPO.

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
        episode_length = ConstantParam(episode_length),
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
    )

    population = set([])
    for _ in range(population_size):
        env = env_generator()
        member = PPOMember(env = env,
                model = create_agent(env),
                hyperparams = generator.generate(),
                use_reward_normalization = use_reward_normalization,
                score = -float('inf'))

        population.add(member)

    return execute_training(population,
            max_rounds=max_rounds,
            round_len=round_len)

