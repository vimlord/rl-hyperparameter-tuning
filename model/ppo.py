
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

        self.policy = Policy(n_inputs, n_outputs, hiddens)
        self.value = Value(n_inputs, hiddens)

    def clone(self):
        other = Agent(self.n_inputs, self.n_outputs, self.hiddens)

        params = self.state_dict()

        params = {
            k : params[k].clone()
            for k in params
        }
        other.load_state_dict(params)

        return other
    
    def forward(self, *args):
        raise NotImplemented

    def encode_state(self, state):
        if self.encoder is None: return state

        state = state.permute(0, 3, 1, 2)
        state = nn.functional.interpolate(state, (84, 84))
        state = self.encoder(state)

        return state

    def choose_action(self, state, randomize=True):
        #state = numpy_to_torch(np.expand_dims(state, 0)).float()
        
        if self.encoder is not None:
            state = self.encode_state(state)
        
        p = self.policy(state)
        
        if randomize:
            # Select from the distribution of actions
            weighting = torch.rand(*p.shape).cuda()
            probs = weighting * p
            a = probs.argmax()#torch.argmax(probs)
        else:
            # Choose the most likely option
            a = p.argmax()

        return a, p[a]

    def step(self, state, randomize=True):
        return self.choose_action(
                state, 
                randomize=randomize)

    def reevaluate(self, states, actions, randomize=True):
        states = self.encode_state(states)

        # Simple: just pass the values in
        V = self.value(states).flatten()
        prob = self.policy(states)
        
        # Generate distributional metadata
        dist = torch.distributions.Categorical(prob)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return V, log_prob, entropy

