
import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs, hiddens=128):
        super().__init__()

        self.fc1 = nn.Linear(n_inputs, hiddens)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hiddens, hiddens)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hiddens, n_outputs)
        self.smax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        y = x

        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        y = self.smax(y)
        
        return y

class Value(nn.Module):
    def __init__(self, n_inputs, hiddens=64):
        super().__init__()

        self.fc1 = nn.Linear(n_inputs, hiddens)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hiddens, hiddens)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hiddens, 1)
    
    def forward(self, x):
        y = x

        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        
        return y

class Quality(nn.Module):
    def __init__(self, n_inputs, n_outputs, hiddens=64):
        super().__init__()

        self.fc1 = nn.Linear(n_inputs, hiddens)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hiddens, hiddens)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hiddens, n_outputs)
    
    def forward(self, x):
        y = x

        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        
        return y

