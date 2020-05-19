
import torch
from torch import optim

import gym

HAS_CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")

def numpy_to_torch(x):
    if isinstance(x, (int, float)): return x
    y = torch.from_numpy(x)
    return y.cuda() if HAS_CUDA else y

def torch_to_numpy(x):
    if isinstance(x, (int, float)): return x
    if HAS_CUDA: x = x.cpu()
    return x.detach().numpy()

class Environment:
    def __init__(self, name='LunarLander-v2'):
        self.env = gym.make(name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        state, reward, done, info = self.env.step(torch_to_numpy(action))
        #done |= info['ale.lives'] != self.lives
        #self.lives = info['ale.lives']
        #print(info)
        return numpy_to_torch(state).float(), reward, float(done)

    def reset(self):
        #self.lives = 5
        return numpy_to_torch(self.env.reset()).float()

    def render(self):
        return self.env.render()

def play_episode(agent, env, render=False):
    done = False
    state = env.reset()

    total_score = 0

    if render: env.render()

    while not done: 
        action = agent.step(state, randomize=False)[0]
        state, reward, done = env.step(action)[:3]
        if render: env.render()
        total_score += reward

    return total_score

def create_simple_optimizer(model, config, Optim=optim.Adam):
    return Optim(model.parameters(), lr=config.lr)


