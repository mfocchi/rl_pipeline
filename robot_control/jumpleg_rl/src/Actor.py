from turtle import forward
import torch 
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim = 256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim,2*layer_dim)
        self.fc3 = nn.Linear(2*layer_dim, layer_dim)
        self.fc4 = nn.Linear(layer_dim, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        action = self.relu(self.fc1(state))
        action = self.relu(self.fc2(action))
        action = self.relu(self.fc3(action))
        action = self.tanh(self.fc4(action))

        return action
