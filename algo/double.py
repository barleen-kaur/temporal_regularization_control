import math
import random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from models.model import DQN, CnnDQN
from utils.replay import ReplayBuffer
from utils.logger import Logger
from utils.loss_plotter import plot, eps_plot


class Double:

    def __init__(self, args, env, device, experiment, _dir):
        self.args = args
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
        self.checkpoint_idx = args.checkpoint_idx
        self.device = device
        self.replay_buffer = ReplayBuffer(args.replay_buff)
        self.experiment =  experiment
        self.log_dir = _dir
        self.env_name = args.env_name.partition("NoFrameskip")
        self.start_frame = args.start_frame
        self.logger = Logger(mylog_path=args.log_dir, mylog_name="training.log", mymetric_names=['frame', 'rewards'])
        
        if args.env_type == "gym":
            self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
            self.target_model  = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        elif args.env_type == "atari":
            self.current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n, self.device)
            self.target_model  = CnnDQN(self.env.observation_space.shape, self.env.action_space.n, self.device)
        if device != "cpu":
            self.current_model = self.current_model.to(self.device)
            self.target_model = self.target_model.to(self.device)
        if args.optim == 'adam':
            self.optimizer = optim.Adam(self.current_model.parameters(), lr=self.args.lr)
        elif args.optim =='rmsprop':
            self.optimizer = optim.RMSprop(self.current_model.parameters(), lr=self.args.lr)


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

        if self.start_frame > 1:
            fr = self.load_checkpoint(self.start_frame)
            print("fr == start_frame: {}".format(fr == self.start_frame))
            print("==> Resuming training from frame: {}".format(self.start_frame))

        no_of_episodes = 0

        for frame_idx in range(self.start_frame, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)
            
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                no_of_episodes += 1
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                self.experiment.log_metric("episode_reward", all_rewards[-1], step=no_of_episodes)
                self.logger.to_csv(np.concatenate((all_rewards[-1], frame_idx)), no_of_episodes)

                
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss() #
                losses.append(loss.item()) 
                self.experiment.log_metric("loss", loss, step=frame_idx)               

            if frame_idx % self.plot_idx == 0:
                plot(frame_idx, all_rewards, losses, self.log_dir, self.env_name[0]) #
                
            if frame_idx % self.target_idx == 0:
                self.update_target()

            if frame_idx % self.checkpoint_idx == 0:
                self.save_checkpoint(frame_idx)


            

    def compute_td_loss(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size) #

        state      = torch.FloatTensor(np.float32(state)).to(self.device) #
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device) #
        action     = torch.LongTensor(action).to(self.device) #
        reward     = torch.FloatTensor(reward).to(self.device) #
        done       = torch.FloatTensor(done).to(self.device) #
        #print("action: {}, shape:{}".format(action, action.shape))
        #print("done: {}, shape:{}".format(done, done.shape))

        q_values = self.current_model(state) #
        next_q_values = self.current_model(next_state) #
        next_q_state_values = self.target_model(next_state) 
        #print("q_values :{}, shape: {}".format(q_values, q_values.shape)) 
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  #
        #print("q_value :{}, shape: {}".format(q_value, q_value.shape))
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1) #
        #print("next_q_value :{}, shape: {}".format(next_q_value, next_q_value.shape))
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        #print("expected_q_value: {}, shape: {}".format(expected_q_value, expected_q_value.shape))
        loss = (q_value - expected_q_value).pow(2).mean() #
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def epsilon_plot(self):
        eps_list = [self.epsilon_by_frame(i) for i in range(self.num_frames)]
        eps_plot(eps_list, self.log_dir, self.env_name[0])


    def save_checkpoint(self, nb_frame):
        w_path = '%s/checkpoint_fr_%d.tar'.format("weights", nb_frame)
        torch.save({
            'frames': fr,
            'modelc': self.current_model.state_dict(),
            'modelt': self.target_model.state_dict()
        }, os.path.join(self.log_dir, w_path))
        print("===> Checkpoint saved to {}".format(w_path))


    def load_checkpoint(self, nb_frame):

        w_path = '%s/checkpoint_fr_%d.tar'.format("weights", nb_frame)
        print("===> Loading Checkpoint saved at {}".format(w_path))
        checkpoint = torch.load(os.path.join(self.log_dir, w_path))
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['modelc'])
        self.target_model.load_state_dict(checkpoint['modelt'])

        return fr




    

