import numpy as np
import rospy
import torch
import torch.nn as nn
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer


class TD3(object):
    def __init__(
        self,
        log_writer,
        state_dim,
        action_dim,
        layer_dim,
        double_critic=True,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr = 1e-4
    ):
        self.double_critic = double_critic
        self.log_writer = log_writer
        self.lr = lr
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TD3 on {self.device}")
        self.actor = Actor(state_dim, action_dim, layer_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, action_dim, double_critic, layer_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.loss_func = nn.MSELoss()
        self.total_it = 0

    def select_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        self.actor.train()
        self.critic.train()

        with torch.no_grad():
                state, action, next_state, reward, done = replay_buffer.sample(batch_size)
                
                noise = (torch.randn_like(action) *
                        self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

                # next_action = torch.nan_to_num(self.actor_target(next_state)+noise).clamp(-1, 1)
                next_action = (self.actor_target(next_state)+noise).clamp(-1, 1)

        if self.double_critic:
            # Compute the target Q value
            target_Q1, target_Q2 =  self.critic_target(next_state, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q
            # print(target_Q)
            # print('target_Q', target_Q)
        
            # Get the current Q estimates
            current_Q1, current_Q2 =  self.critic(state, action)

            # Compute critic loss
            critic_loss = self.loss_func(current_Q1, target_Q) + self.loss_func(current_Q2, target_Q)

        else:
            target_Q = reward
            
            current_Q1 =  self.critic(state, action)

            # Compute critic loss
            critic_loss = self.loss_func(current_Q1, target_Q)
            

        self.log_writer.add_scalar("Critic loss", critic_loss.item(), self.total_it)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optimizer.step()

        # Delayed policy updates

        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            
            # actor_loss = - self.critic.Q1(state, nn.functional.normalize(self.actor(state),2)).mean()
            actor_loss = - self.critic.Q1(state, self.actor(state)).mean()
            # print('actor loss',actor_loss.item())
            # rospy.loginfo("Updating the networks")
            self.log_writer.add_scalar("Actor loss", actor_loss.item(), self.total_it)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path, name):
        torch.save(self.critic.state_dict(), os.path.join(
            path, f"TD3_{name}_critic.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            path, f"TD3_{name}_critic_optimizer.pt"))

        torch.save(self.actor.state_dict(), os.path.join(
            path, f"TD3_{name}_actor.pt"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            path, f"TD3_{name}_actor_optimizer.pt"))

    def load(self, path, name, iteration):
        self.total_it = iteration
        
        self.critic.load_state_dict(torch.load(
            os.path.join(path, f"TD3_{name}_critic.pt")))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(path, f"TD3_{name}_critic_optimizer.pt")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(
            os.path.join(path, f"TD3_{name}_actor.pt")))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(path, f"TD3_{name}_actor_optimizer.pt")))
        self.actor_target = copy.deepcopy(self.actor)