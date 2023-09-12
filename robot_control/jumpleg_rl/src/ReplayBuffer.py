import numpy as np
import torch
import joblib
import os


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.index = 0
        self.mem_size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def get_number_episodes(self):
        return self.index

    def store(self, state, action, next_state, reward, done):
        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward
        self.done[self.index] = done

        self.index = (self.index + 1) % self.max_size
        self.mem_size = min(self.mem_size + 1, self.max_size)

    def sample(self, batch_size):
        random_index = np.random.randint(0, self.mem_size, size=batch_size)

        return (
            torch.FloatTensor(self.state[random_index]).to(self.device),
            torch.FloatTensor(self.action[random_index]).to(self.device),
            torch.FloatTensor(self.next_state[random_index]).to(self.device),
            torch.FloatTensor(self.reward[random_index]).to(self.device),
            torch.FloatTensor(self.done[random_index]).to(self.device)
        )

    def dump(self, out_path, agent_mode):
        joblib.dump(self, os.path.join(out_path,f'ReplayBuffer_{agent_mode}.joblib'))