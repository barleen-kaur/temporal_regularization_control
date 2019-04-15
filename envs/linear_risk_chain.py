"""
Environment where you have a linear states that's risky. Wrong action and you get to the start with a negative reward. 

Written by modifying y_chain.py
"""


import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

class LinearRiskChain():
    """
    """
    def __init__(self, n=10):

        # Initialzing required parameters
        self.update_count = 0
        self.n = n # Length of the chain
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2) # Number of actions: 2 - [0: step forward, 1: reset]
        self.step_reward = 0
        self.neg_reward = -1
        self.observation_space = spaces.Discrete(self.n+1) # number of states is equal to chain length
        self.seed() # not sure what this does, so not changing it

        # Saving optimal q values (for eps = 0)
        self.optimalQ = np.column_stack((self.step_reward * np.hstack((np.ones(n),0)), self.neg_reward * np.hstack((np.ones(n),0))))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        # Making sure valid action is chosen
        assert self.action_space.contains(action)

        self.update_count += 1

        # Stepping along on the chain
        if(action == 0):
            self.state = self.state + 1
            reward = self.step_reward
        else: # resetting the state
            self.state = 0
            reward = self.neg_reward

        # Because this is a continuing case
        if(self.state == self.n):
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.update_count = 0 
        self.state = 0
        return self.state
    
# register(
#     id='Ring-v0',
#     entry_point='ychain:YChain',
#     timestep_limit=20000,
#     reward_threshold=1,
# )




