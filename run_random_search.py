
import torch
import gym

from matplotlib import pyplot as plt

from util import Environment, play_episode

import random_search.train.ppo
import random_search.train.dqn

import torch.multiprocessing as mp

import argparse

from random_search.train import train

if __name__ == '__main__':
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(
            description='Trains an RL agent using random search'
    )

    parser.add_argument('--env-name',
            default='LunarLander-v2',
            help='The environment to use')

    parser.add_argument('--episode-length',
            default=2048,
            type=int,
            help='The maximum number of steps to evaluate in an episode')

    parser.add_argument('--max-epochs',
            default=1024,
            type=int,
            help='The number of rounds to run per population member')

    parser.add_argument('--max-attempts',
            default=4,
            type=int,
            help='The number of models to train')


    parser.add_argument('--max-processes',
            default=mp.cpu_count(),
            type=int,
            help='The max number of models to train at once')

    parser.add_argument('--algo',
            default='ppo',
            help='The algorithm to use')

    parser.add_argument('--use-reward-normalization',
            type=bool,
            default=True,
            help='Whether or not to normalize the rewards')

    args = parser.parse_args()

    env_name = args.env_name
    algo = args.algo

    module = {
        'ppo' : random_search.train.ppo,
        'dqn' : random_search.train.dqn,
    }[algo]

    print('Algorithm:', algo)
    print('Environment:', env_name)
    print('Process count:', args.max_processes)

    env = Environment(env_name)

    def model_trial(agent):
        global best_score
        global best_agent

        n_episodes = 30
        score = sum(play_episode(agent, env) for _ in range(n_episodes)) / n_episodes
        
        if score > best_score:
            print('Best score so far is', score)
            best_agent, best_score = agent, score

    best_score = -float('inf')
    best_agent = None

    fn_args = {
        "env_name" : args.env_name,
        "episode_length" : args.episode_length,
        "use_reward_normalization" : args.use_reward_normalization
    }

    if args.max_processes == 1:
        for _ in range(args.max_attempts):
            agent = module.train(**fn_args)
            model_trial(agent)
    else:
        ctx = mp.get_context('spawn')

        pool = ctx.Pool(args.max_processes)

        processes = [
                pool.apply_async(train, (module.train, fn_args))
                for _ in range(args.max_attempts)
        ]

        weightings = [p.get() for p in processes]

        pool.close()

        for params in weightings:
            agent = module.create_agent(env)
            agent.load_state_dict(params)
            model_trial(agent)
    
    print('An optimal model has been chosen')
    torch.save(agent.cpu(), 'model.pt')

