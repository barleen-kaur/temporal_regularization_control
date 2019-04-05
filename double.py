import math
import random
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from models.model import DQN, CnnDQN
from utils.replay import ReplayBuffer
from utils.loss_plotter import plot



USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Double:

    def __init__(self, args, env, device):
        self.env_type = args.env_type
        self.env = env
        self.eps_s = args.eps_s
        self.eps_f = args.eps_f
        self.eps_decay = args.eps_decay
        self.num_frames = args.num_frames
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.plot_idx = args.plot_idx
        self.target_idx = args.target_idx
        self.replay_buffer = ReplayBuffer(args.buff_size)
        if args.env_type == "gym":
            self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
            self.target_model  = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        elif args.env_type == "atari":
            self.current_model = CnnDQN(self.env.observation_space.shape[0], self.env.action_space.n)
            self.target_model  = CnnDQN(self.env.observation_space.shape[0], self.env.action_space.n)
        if device != "cpu":
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()
        if args.optim == 'Adam':
            self.optimizer = optim.Adam(self.current_model.parameters(), lr=args.lr)
        elif args.optim =='rmsprop':
            self.optimizer = optim.RMSprop(self.current_model.parameters(), lr=args.lr)


    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())


    def epsilon_by_frame(self, frame_idx):
        return self.eps_f + (self.eps_s - self.eps_f) * math.exp(-1. * frame_idx / self.eps_decay)


    def train(self):

        losses = []
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()

        self.update_target()

        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)
            
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss() #
                losses.append(loss.item())
                
            if frame_idx % self.plot_idx == 0:
                plot(frame_idx, all_rewards, losses) #
                
            if frame_idx % self.target_idx == 0:
                self.update_target()


    def compute_td_loss(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size) #

        state      = Variable(torch.FloatTensor(np.float32(state))) #
        next_state = Variable(torch.FloatTensor(np.float32(next_state))) #
        action     = Variable(torch.LongTensor(action)) #
        reward     = Variable(torch.FloatTensor(reward)) #
        done       = Variable(torch.FloatTensor(done)) #

        q_values      = current_model(state) #
        next_q_values = current_model(next_state) #
        next_q_state_values = target_model(next_state)  

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  #
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1) #
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean() #
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss




plt.plot([epsilon_by_frame(i) for i in range(10000)])

    

