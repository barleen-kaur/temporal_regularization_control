import os
import math
import time
import random
import numpy as np
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from models.model import Deep, CnnDQN, LinearFA
from utils.replay import ReplayBuffer
from utils.logger import Logger 
from utils.loss_plotter import eps_plot, LossPlotter


class DQN:

    def __init__(self, args, env, env_name, device, experiment, _dir):
        self.args = args
        self.env_type = args.env_type
        self.eps_s = args.eps_s
        self.eps_f = args.eps_f
        self.eps_decay = args.eps_decay
        self.num_frames = args.num_frames
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.plot_idx = args.plot_idx
        self.target_idx = args.target_idx
        self.checkpoint_idx = args.checkpoint_idx
        self.env_name = env_name
        self.start_frame = args.start_frame
        self._beta = args.beta
        self._lambda = args.lamb
        self.env = env
        self.device = device
        self.replay_buffer = ReplayBuffer(args.replay_buff)
        self.replay_threshold = args.replay_threshold
        self.experiment =  experiment
        self.log_dir = _dir
        self.action_count = 0
        self.episode_rewards = deque(maxlen=self.args.return_deque)
        self._p = torch.zeros([1, self.env.action_space.n], dtype=torch.float32)
        self.logger = Logger(mylog_path=self.log_dir, mylog_name=self.env_name+"_"+self.args.FA+"_training.log", mymetric_names=['frame', 'episodes_done' , 'episode_return', 'loss', 'action_change'])
        self.LP = LossPlotter(mylog_path=self.log_dir, mylog_name=self.env_name+"_"+self.args.FA+"_training.log", env_name=self.env_name+"_"+self.args.FA, xmetric_name= 'frame', ymetric_names=['episode_return', 'loss', 'action_change'])


        if self.args.env_type == "gym" and self.args.FA == "linear":
            self.model = LinearFA(self.env.observation_space.shape[0], self.env.action_space.n, self.device)
            
        if self.args.env_type == "gym" and self.args.FA == "deep":
            self.model = Deep(self.env.observation_space.shape[0], self.env.action_space.n, self.device)
            
        elif self.args.env_type == "atari":
            self.model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n, self.device)
            
        if device != "cpu":
            self.model = self.model.to(self.device)
            
        if self.args.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim =='rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.lr)



    def epsilon_by_frame(self, frame_idx):
        return self.eps_f + (self.eps_s - self.eps_f) * math.exp(-1. * frame_idx / self.eps_decay)


    def train(self):

        #Initializations
        losses = []
        _return = 0
        state = self.env.reset()
        previous_action = None
        no_of_episodes = 0

        
        #Getting p for this state
        state_unsqueezed = torch.FloatTensor(np.float32(state)).unsqueeze_(0).to(self.device)
        self._p = self.model(state_unsqueezed)
        #print("_p.shape: {}".format(self._p.shape))


        #Loading previous saved parameters incase of resuming the experiment
        if self.start_frame > 1:
            no_of_episodes = self.load_checkpoint(self.start_frame)
            print("==> Resuming training from frame: {}".format(self.start_frame)) 
        
        #start_time = time.time()
        for frame_idx in range(self.start_frame+1, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            action = self.model.act(state, epsilon)
            p_action = self._p[0][action].detach()
            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done, p_action)
            
            state = next_state
            _return += reward

            if previous_action != None and previous_action != action:
                self.action_count = self.action_count + 1

            previous_action = action

            if done:
                no_of_episodes += 1
                #print("No of episodes ended: {}".format(no_of_episodes))
                state = self.env.reset()
                self.episode_rewards.append(_return) 
                _return = 0
                previous_action = None
                state_unsqueezed = torch.FloatTensor(np.float32(state)).unsqueeze_(0).to(self.device)
                self._p = self.model(state_unsqueezed)

                
            if len(self.replay_buffer) > self.replay_threshold:
                loss = self.compute_td_loss() #
                losses.append(loss.item()) 
            
            #self.experiment.log_metric("loss", loss, step=frame_idx)               

            if frame_idx  % self.plot_idx == 0:
                #print("Frame: {}, Reward: {}, Loss: {}, action: {}".format(frame_idx, np.mean(self.episode_rewards), np.mean(losses), self.action_count))
                self.logger.to_csv(np.array([frame_idx, no_of_episodes , np.mean(self.episode_rewards), np.mean(losses), self.action_count]), self.plot_idx)
                self.LP.plotter() 
                self.experiment.log_metric("loss", np.mean(losses), step=frame_idx)
                self.experiment.log_metric("return", np.mean(self.episode_rewards), step=frame_idx)
                self.experiment.log_metric("action", self.action_count, step=frame_idx)
                losses =[]
                self.action_count = 0
                #print("{} frames done in {} sec".format(self.plot_idx, time.time()-start_time))
                #start_time = time.time()

            if frame_idx % self.checkpoint_idx == 0:
                self.save_checkpoint(frame_idx, no_of_episodes)
            
            
            state_unsqueezed = torch.FloatTensor(np.float32(state)).unsqueeze_(0).to(self.device)
            q_value = self.model(state_unsqueezed)
            self._p = (1- self._lambda)*q_value + self._lambda*self._p
            

    def compute_td_loss(self):

        
        state, action, reward, next_state, done, p_action = self.replay_buffer.sample(self.batch_size) 
        
        state      = torch.FloatTensor(np.float32(state)).to(self.device)  
        action     = torch.LongTensor(action).to(self.device) 
        reward     = torch.FloatTensor(reward).to(self.device) 
        done       = torch.FloatTensor(done).to(self.device)
        p_action  = torch.FloatTensor(p_action).to(self.device)
        
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        #print("action: {}, shape:{}".format(action, action.shape))
        #print("done: {}, shape:{}".format(done, done.shape))

        q_values = self.model(state) 
        next_q_values = self.model(next_state).detach()  
        #print("q_values :{}, shape: {}".format(q_values, q_values.shape)) 
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  
        #print("q_value :{}, shape: {}".format(q_value, q_value.shape))
        next_q_value = next_q_values.max(1)[0]
        #print("next_q_value :{}, shape: {}".format(next_q_value, next_q_value.shape))
        expected_q_value = reward + self.gamma *((1.0-self._beta)*(1-done)*next_q_value + self._beta*p_action)
        #print("expected_q_value: {}, shape: {}".format(expected_q_value, expected_q_value.shape))
     
        loss = (q_value - expected_q_value).pow(2).mean() 
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss



    def epsilon_plot(self):
        eps_list = [self.epsilon_by_frame(i) for i in range(self.num_frames)]
        eps_plot(eps_list, self.log_dir, self.env_name+"_"+self.args.FA)


    def save_checkpoint(self, nb_frame, epi):
        w_path = '%s/checkpoint_fr_%d.tar'%(self.env_name+"_"+self.args.FA+"_weights", nb_frame)
        torch.save({
            'frames': nb_frame,
            'epi': epi,
            'deque': self.episode_rewards,
            'action': self.action_count,
            'buffer': self.replay_buffer,
            'model': self.model.state_dict(),
        }, os.path.join(self.log_dir, w_path))
        #print("===> Checkpoint saved to {}".format(w_path))


    def load_checkpoint(self, nb_frame):

        w_path = '%s/checkpoint_fr_%d.tar'%(self.env_name+"_"+self.args.FA+"_weights", nb_frame)
        print("===> Loading Checkpoint saved at {}".format(w_path))
        checkpoint = torch.load(os.path.join(self.log_dir, w_path))
        fr = checkpoint['frames']
        epi = checkpoint['epi']
        self.deque = checkpoint['deque']
        self.action_count = checkpoint['action']
        self.replay_buffer = checkpoint['buffer']
        self.model.load_state_dict(checkpoint['model'])

        return epi




    

