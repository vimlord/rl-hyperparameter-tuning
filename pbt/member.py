
import random

import torch.multiprocessing as mp
import threading

from itertools import product

from util import play_episode, create_simple_optimizer

class Member:
    def __init__(self, model, hyperparams, score, procedure, env):
        self.model = model
        self.hyperparams = hyperparams
        self.score = score
        self.env = env
        self.procedure = procedure

        self.reset_optimizer()
    
    def reset_optimizer(self):
        # Apply optimizer
        optimizer = create_simple_optimizer(
                model = self.model,
                config = self.hyperparams)

        self.hyperparams.optimizer = self.optimizer = optimizer

    def clone(self):
        return type(self)(
                model=self.model.clone(),
                hyperparams=self.hyperparams.clone(),
                score=self.score,
                procedure=self.procedure,
                env=self.env
        ) 

    def value_of(self, key):
        return self.hyperparams[key]

    def move_model_to_cpu():
        self.model.cpu()

    def move_model_to_cuda():
        self.model.cuda()

    def explore(self):
        self.hyperparams = self.hyperparams.mutate()
        self.reset_optimizer()

    def replace(self, src):
        state_dict = src.model.state_dict()
        self.model.load_state_dict({
            k : state_dict[k].clone()
            for k in state_dict
        })
        self.hyperparams = src.hyperparams.copy()
        self.score = src.score
        
        # Reset the optimizer to be compatible with the model
        self.reset_optimizer()

    def evaluate(self, update=True, **args):
        if update:
            self.score = self.compute_score(**args)

        return self.score

    def compute_score(self, n_episodes=5):
        """ Evaluates the model, by default playing some episodes
        and averaging the scores.
        """

        # Score is computed via aggregate over multiple episodes
        score = 0

        for _ in range(n_episodes):
            score += play_episode(self.model, self.env)

        return score / n_episodes

