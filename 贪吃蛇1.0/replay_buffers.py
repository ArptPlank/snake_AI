import collections
import random
from torch import FloatTensor
import torch
import numpy as np
class ReplayBuffer():
    def __init__(self,max_size,num_steps=4,device='cuda'):
        self.buffer = collections.deque(maxlen=max_size)
        self.num_steps = num_steps
        self.device = device

    def append(self,exp):
        self.buffer.append(exp)

    def sample(self,batch_size):
        mini_batch = random.sample(self.buffer,batch_size)
        obs_batch,action_batch,reward_batch,next_obs_batch,done_batch = zip(*mini_batch)
        # obs_batch = torch.FloatTensor(obs_batch)
        # action_batch = torch.FloatTensor(action_batch)
        # reward_batch = torch.FloatTensor(reward_batch)
        # next_obs_batch = torch.FloatTensor(next_obs_batch)
        # done_batch = torch.FloatTensor(done_batch)
        obs_batch = torch.stack([torch.tensor(x, device=self.device) for x in obs_batch])
        action_batch = torch.stack([torch.tensor(x, device=self.device) for x in action_batch])
        reward_batch = torch.stack([torch.tensor(x, device=self.device) for x in reward_batch])
        next_obs_batch = torch.stack([torch.tensor(x, device=self.device) for x in next_obs_batch])
        done_batch = torch.stack([torch.tensor(x, device=self.device) for x in done_batch])
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch
#, dtype=torch.float32
    def __len__(self):
        return len(self.buffer)