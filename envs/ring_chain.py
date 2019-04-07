"""
Environment where you have a linear ring of states.

Written by modifying y_chain.py
"""


import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

class RingChain():
    """
    Description:
        A chain of length n. If you move forward (action 0), you get step_reward and go to the next state. If you taken action 1, you get neg_reward and fall to the state 0. 

    Episode termination:
        There's no termination, this is a continuing environment.

    Usage: 
        from ring_chain import RingChain
        env = Ring()
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
    """
    def __init__(self, n=10):
        self.n = n # Length of the chain
        self.state = 0  # Start at beginning of the ring
        self.action_space = spaces.Discrete(2) # Number of actions: 2 - [0: step forward, 1: reset]
        self.step_reward = 0
        self.neg_reward = -1
        self.observation_space = spaces.Discrete(self.n) # number of states is equal to chain length
        self.seed() # not sure what this does, not changing it

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        # Making sure valid action is chosen
        assert self.action_space.contains(action)

        # Stepping along on the chain
        if(action == 0):
            self.state = (self.state + 1) % self.n
            reward = self.step_reward
        else: # resetting the state
            self.state = 0
            reward = self.neg_reward

        # Because this is a continuing case
        done = False

        return self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.state = 0
        return self.state
    
# register(
#     id='Ring-v0',
#     entry_point='ychain:YChain',
#     timestep_limit=20000,
#     reward_threshold=1,
# )




