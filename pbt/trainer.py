
from tqdm import trange, tqdm

import numpy as np
import random

import torch.multiprocessing as mp
import threading

from itertools import product

def parallel_train(trainer, member, args):
    trainer.train_member_once(member, **args)

def parallel_train_fully(trainer, member, args):
    trainer.train_member_completely(member, **args)

class Trainer:
    def __init__(self, population=None, percentiles=(50,50)):
        if population is None:
            self.agents = set([])
            self.population = set([])
        else:
            self.agents = set(p for p in population)
            self.population = set([])
            for p in self.agents:
                self.add_population_member(p.clone())

        self.score_history = []

        self.percentiles = percentiles

    def exploit(self, member):
        """ Uses a more optimal model if the current model is subpar.
        """
        scores = [m.evaluate(update=False) for m in self.population]
        score = member.score

        lo, hi = np.percentile(scores, self.percentiles)
        
        if score > lo:
            return False
 
        # Choose one of the top tier models
        members = [m for m in self.population if m.evaluate(update=False) >= hi]
        alternate = random.choice(members)

        # Replace with the alternate model
        member.replace(alternate)

        return True

    def step(self, member):
        member.procedure.do_single_training_round(
                member.model, member.hyperparams)

    def add_population_member(self, member):
        self.population.add(member)
        member.model.cpu()
    
    def train(self, max_rounds=1000, n_threads=1, **args):
        """ Do complete training for all configurations.
        To introduce parallelization, create threads that
        each call train_member_completely.
        """
        if n_threads >= len(self.agents):
            args['max_rounds'] = max_rounds
            threads = []
            for p in self.agents:
                t = threading.Thread(target=parallel_train_fully,
                        args=(self, p, args))
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            rng = trange(max_rounds, desc='Starting training')

            for _ in rng:
                self.train_all_members_once(**args)
                rng.set_description(f'Best score: {max(p.score for p in self.population):.4f}')

    def train_all_members_once_parallel(self, n_threads=4, **args):
        agents = list(self.agents)
        
        random.shuffle(agents)
        threads = []
        for p in agents:
            # Create a thread
            t = threading.Thread(
                    target=parallel_train,
                    args=(self, p, args))
            threads.append(t)
        
        # Handle spawning
        if n_threads >= len(agents):
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for i in range(n_threads):
                threads[i].start()
            for i, t in enumerate(threads):
                t.join
                i += n_threads
                if i < len(threads):
                    threads[i].start()

    def train_all_members_once_sequential(self, **args):
        agents = list(self.agents)
        random.shuffle(agents)

        #agents = tqdm(agents, desc='Single round')

        for member in agents:
            self.train_member_once(member)

    def train_all_members_once(self, n_threads=1, **args):
        """ Single wave of training for all population members.
        """
        if n_threads > 1:
            self.train_all_members_once_parallel(n_threads=n_threads, **args)
        else:
            self.train_all_members_once_sequential(**args)

        self.score_history.append(max(p.score for p in self.population))

    def attempt_explore_exploit(self, member):
        changed = self.exploit(member)
        
        if changed:
            # Permute the hyperparams
            member.explore()
        else:
            # Provide self as a population member
            self.add_population_member(member.clone())

    def train_member_once(self, member, **args):
        # Do training
        self.train_member_round(member, **args)

        # Attempt to exploit for better model/hyperparams
        self.attempt_explore_exploit(member)

    def train_member_completely(self, member, max_rounds=1000, **args):
        rng = trange(max_rounds, desc='Training thread')

        #for _ in range(max_rounds):
        for _ in rng:
            self.train_member_once(member, **args)
            rng.set_description(f'Best score: {max(p.score for p in self.population):.4f}')


    def train_member_round(self, member, round_len=32, eval_rate=1):
        assert round_len % eval_rate == 0

        # Do training
        for t in range(round_len):
        #for t in range(round_len, desc='Single agent round'):
            # Model step
            self.step(member)

            # Evaluation
            if (t+1) % eval_rate == 0:
                member.evaluate(update=True)

