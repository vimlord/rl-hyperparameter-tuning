
import torch
import gym

from matplotlib import pyplot as plt

from util import Environment

import pbt.train.ppo
import pbt.train.dqn

import argparse

from torch import multiprocessing as mp

parser = argparse.ArgumentParser(
        description='Trains an RL agent using Population-Based Training'
)

parser.add_argument('--env-name',
        default='LunarLander-v2',
        help='The environment to use')

parser.add_argument('--episode-length',
        default=2048,
        type=int,
        help='The maximum number of steps to evaluate in an episode')

parser.add_argument('--max-rounds',
        default=1024,
        type=int,
        help='The number of rounds to run per population member')


parser.add_argument('--num-threads',
        default=mp.cpu_count(),
        type=int,
        help='The number of threads to use')

parser.add_argument('--round-len',
        default=8,
        type=int,
        help='The number of rounds to run per population member')

parser.add_argument('--population-size',
        default=4,
        type=int,
        help='The number of models to train in parallel')

parser.add_argument('--algo',
        default='ppo',
        help='The algorithm to use')

parser.add_argument('--use-reward-normalization',
        type=bool,
        default=True,
        help='Whether or not to normalize the rewards')

args = parser.parse_args()

# Environment definition
env_name = args.env_name

training_module = {
    'ppo' : pbt.train.ppo,
    'dqn' : pbt.train.dqn
}[args.algo]

print('Algorithm:', args.algo)
print('Environment:', env_name)

trainer = training_module.train(lambda: Environment(env_name),
        num_threads=args.num_threads,
        episode_length=args.episode_length,
        max_rounds=args.max_rounds,
        round_len=args.round_len,
        population_size=args.population_size,
)

best_score = max(p.score for p in trainer.population)
best_members = [p for p in trainer.population if p.score == best_score]

print('Found', len(best_members), 'candidates for best model')

hst = trainer.score_history
t = list(range(1, len(hst)+1))

# Save the models
for i, mdl in enumerate(best_members):
    torch.save(mdl.model.cpu(), f'model{i+1}.pt')

plt.plot(t, hst)
plt.show()

