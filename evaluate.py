
import torch

from util import Environment, play_episode

import gym

import argparse

parser = argparse.ArgumentParser(
        description='Trains an RL agent using Population-Based Training'
)

parser.add_argument('--agent',
        help='The filename of the agent to run')

parser.add_argument('--env-name',
        default='LunarLander-v2',
        help='The environment to use')

args = parser.parse_args()

env_name = args.env_name
mdl_name = args.mdl_name

env = Environment(env_name)
agent = torch.load(mdl_name).cuda()

while True:
    score = play_episode(agent, env, render=True)
    print(score)


