
import torch
from torch import nn

from util import *

import random

from model import *

class Agent(nn.Module):
    def __init__(self, n_inputs, n_outputs, hiddens=64):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hiddens = hiddens

        self.encoder = None
        
        self.Q = Quality(n_inputs, n_outputs, hiddens)
        self.Q_tgt = Quality(n_inputs, n_outputs, hiddens)

    def clone(self):
        other = Agent(self.n_inputs, self.n_outputs, self.hiddens)

        params = self.state_dict()

        params = {
            k : params[k].clone()
            for k in params
        }
        other.load_state_dict(params)

        return other

    def update_target(self, tau=1e-3):
        src = self.Q.state_dict()
        tgt = self.Q_tgt.state_dict()

        dct = {
            k: tau * src[k] + (1-tau) * tgt[k]
            for k in src
        }

        self.Q_tgt.load_state_dict(dct)
        
    def quality(self, state, action=None, target=False):
        Q = self.Q_tgt if target else self. Q
        
        Q = Q(state)
        
        if action is None:
            return Q
        else:
            idxs = torch.zeros_like(Q).long()
            if HAS_CUDA: idxs = idxs.cuda()

            idxs[:,0] = action.long()

            return Q.gather(1, idxs)[:,0]

    def value(self, state, target=False):
        return self.quality(state, target=target).max(dim=-1)[0]
    
    def choose_action(self, state, randomize=0.1):
        if randomize and random.random() < randomize:
            if len(state.shape) == 1:
                A = torch.tensor(random.randrange(0, self.n_outputs))
                if HAS_CUDA: A = A.cuda()

                return A
            else:
                return numpy_to_torch(
                        np.array([
                                random.randrange(0, n_outputs)
                                for _ in range(state.shape[0])]
                ))
        else:
            Q = self.quality(state)
            A = Q.argmax().cpu().int()
            return A

    def step(self, state, randomize=0.1):
        return (self.choose_action(state, randomize=randomize),)

