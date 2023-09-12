from turtle import forward
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, double_critic = True, layer_dim = 256):
        super(Critic, self).__init__()

        self.double_critic = double_critic

        # Q1 critic    
        self.q1_fc1 = nn.Linear(state_dim+action_dim, layer_dim)
        self.q1_fc2 = nn.Linear(layer_dim,2*layer_dim)
        self.q1_fc3 = nn.Linear(2*layer_dim, layer_dim)
        self.q1_fc4 = nn.Linear(layer_dim, 1)

        if self.double_critic:
            # Q2 critic
            self.q2_fc1 = nn.Linear(state_dim+action_dim, layer_dim)
            self.q2_fc2 = nn.Linear(layer_dim,2*layer_dim)
            self.q2_fc3 = nn.Linear(2 * layer_dim, layer_dim)
            self.q2_fc4 = nn.Linear(layer_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        sa_cat = torch.cat([state, action], 1)

        q1 = self.relu(self.q1_fc1(sa_cat))
        q1 = self.relu(self.q1_fc2(q1))
        q1 = self.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)

        if self.double_critic:

            q2 = self.relu(self.q2_fc1(sa_cat))
            q2 = self.relu(self.q2_fc2(q2))
            q2 = self.relu(self.q2_fc3(q2))
            q2 = self.q2_fc4(q2)

            return q1, q2
        
        else:
            
            return q1
    
    def Q1(self, state, action):
        sa_cat = torch.cat([state, action], 1)
    
        q1 = self.relu(self.q1_fc1(sa_cat))
        q1 = self.relu(self.q1_fc2(q1))
        q1 = self.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)
    
        return q1
