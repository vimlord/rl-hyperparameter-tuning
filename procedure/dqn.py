
import torch
from torch import nn
from torch import optim

import numpy as np

import random

from procedure import TrainingProcedure

import threading

from util import numpy_to_torch, HAS_CUDA

class MemoryBuffer:
    def __init__(self, dims=5, max_size=10000):
        self.buffs = [[] for _ in range(dims)]
        self.max_size = max_size
        self.idx = 0

        self.lock = threading.Lock()

    def __len__(self):
        return len(self.buffs[0])

    def add(self, *xs):
        with self.lock:
            if len(self.buffs[0]) == self.max_size:
                for buff, x in zip(self.buffs, xs):
                    buff[self.idx] = x
                    self.idx = (self.idx + 1) % self.max_size
            else:
                for buff, x in zip(self.buffs, xs):
                    buff.append(x)

            assert all(len(b) == len(self.buffs[0]) for b in self.buffs)

    def sample(self, n_samples=32):
        idxs = list(range(len(self.buffs[0])))
        idxs = random.sample(idxs, n_samples)

        with self.lock:
            data = []
            for j, b in enumerate(self.buffs):
                d = [b[i] for i in idxs]

                if isinstance(d[0], (float, int, np.float64)) or len(d[0].shape) == 0:
                    #data.append(numpy_to_torch(np.array(d)))
                    d = torch.tensor(d)
                    if HAS_CUDA: d = d.cuda()
                    data.append(d)
                else:
                    data.append(torch.stack(d))

            return data

class DQNTrainingProcedure(TrainingProcedure):
    def __init__(self, mem_buff, use_reward_normalization=False, **args):
        super().__init__(**args)
        self.mem_buff = mem_buff
        self.use_reward_normalization = use_reward_normalization

        self.rand_rate = 1.
        self.rand_decay = 0.999
        self.rand_min = 0.1

    def update_rand(self):
        self.rand_rate = max(self.rand_rate * self.rand_decay, self.rand_min)

    def do_single_training_round(self, agent, config):
        done = 0.
        env = self.env

        state = env.reset()

        while done < 0.5:
            action = agent.choose_action(state, randomize=self.rand_rate)

            next_state, reward, done = env.step(action)

            self.mem_buff.add(
                    state, action, next_state, reward, done)


            if len(self.mem_buff) >= config.batch_size:
                self.do_gradient_step(agent, config)

            state = next_state
        
        self.update_rand()
    
    def do_gradient_step(self, agent, config):
        S0, A, S1, R, D = self.mem_buff.sample(config.batch_size)

        R = (R - R.mean()) / (R.std() + 1e-6)

        gamma = config.gamma
        optimizer = config.optimizer

        optimizer.zero_grad()

        Q = agent.quality(S0, A)
        V = agent.value(S1, target=True).detach()
        Q_bellman = R + gamma * (1-D) * V

        loss = config.loss_fn(Q_bellman, Q)
        
        # Perform optimization
        loss.backward()
        optimizer.step()
        
        agent.update_target(tau=config.tau)

