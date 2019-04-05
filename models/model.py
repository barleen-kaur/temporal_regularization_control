import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F


class DQN(nn.Module):
    
    def __init__(self, inp_channel, out_channel):
        super(DQN, self).__init__()

        self.inp_channel = inp_channel
        self.out_channel = out_channel
        
        self.layers = nn.Sequential(
            nn.Linear(self.inp_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_channel)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                state   = torch.FloatTensor(state).unsqueeze(0) #change this
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
            else:
                action = random.randrange(self.out_channel)
        return action


class CnnDQN(nn.Module):

    def __init__(self, inp_shape, out_channel, device):
        super(CnnDQN, self).__init__()
        
        self.inp_shape = inp_shape
        self.out_channel = out_channel
        self.device = device
        
        self.features = nn.Sequential(
            nn.Conv2d(self.inp_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.out_channel)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.inp_shape).to(self.device)).view(1, -1).size(1) #change this
    
    def act(self, state, epsilon):

        with torch.no_grad():
            if random.random() > epsilon:
                state   = torch.FloatTensor(np.float32(state)).unsqueeze(0) #change this
                q_value = self.forward(state.to(self.device))
                action  = q_value.max(1)[1].item()
            else:
                action = random.randrange(self.out_channel)
        return action
