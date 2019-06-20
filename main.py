

import os
import time
import random
import numpy as np
from collections import deque
from comet_ml import OfflineExperiment, Experiment

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from arguments import get_args
from algo.double import Double
from algo.dqn import DQN

args = get_args()


assert args.algo in ['dqn', 'double']

if args.comet == "offline":
    experiment = OfflineExperiment(project_name="recurrent-value", workspace="nishanthvanand",disabled=args.disable_log, offline_directory="../comet_offline", parse_args=False)
elif args.comet == "online":
    experiment = Experiment(api_key="piz8lnPU3GY0GPZQ10mef0rdU", project_name="temporal-reg-control", workspace="barleenk", disabled=args.disable_log, parse_args=False)
else:
    raise ValueError

experiment.log_parameters(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



#########################################################################################################

def _env_set(env_name, env_type):

    if env_type == 'gym':
        env = gym.make(env_name)

    elif env_type == 'atari':
        env = make_atari(env_name, skip=args.frame_skip)
        env = wrap_deepmind(env, frame_stack=args.frame_stack)
        env = wrap_pytorch(env)

    return env

def _main():
    
    device = torch.device("cuda" if args.cuda else "cpu")


    if args.algo == 'dqn':
        #figure out directory stuff
        _dir = os.path.join(args.log_dir, "dqn_results")
        if args.env_type == "atari":
           env_n = args.env_name.partition("NoFrameskip")
           os.makedirs(_dir, exist_ok=True) #change this
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb)), exist_ok=True)
           _dir = os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb))
           os.makedirs(os.path.join(_dir, env_n[0]+"_"+args.FA+'_weights'), exist_ok=True)
        elif args.env_type == "gym":
           env_n = args.env_name.partition("-")
           os.makedirs(_dir, exist_ok=True) #change this
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb)), exist_ok=True)
           _dir = os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb))
           os.makedirs(os.path.join(_dir, env_n[0]+"_"+args.FA+'_weights'), exist_ok=True)
        
        env = _env_set(args.env_name, args.env_type)
        env.seed(args.seed) 
        alg = DQN(args, env, env_n[0], device, experiment, _dir)
        alg.epsilon_plot()
        alg.train()

    elif args.algo == 'double':
        #figure out directory stuff
        _dir = os.path.join(args.log_dir, "double_results")
        if args.env_type == "atari":
           env_n = args.env_name.partition("NoFrameskip")
           os.makedirs(_dir, exist_ok=True) #change this 
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb)), exist_ok=True)
           _dir = os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb))
           os.makedirs(os.path.join(_dir, env_n[0]+"_"+args.FA+'_weights'), exist_ok=True)
        elif args.env_type == "gym":
           env_n = args.env_name.partition("-")
           os.makedirs(_dir, exist_ok=True) #change this
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta)), exist_ok=True)
           os.makedirs(os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb)), exist_ok=True)
           _dir = os.path.join(_dir,env_n[0],'seed_'+str(args.seed),'beta_'+str(args.beta),'lambda_'+str(args.lamb))
           os.makedirs(os.path.join(_dir, env_n[0]+"_"+args.FA+'_weights'), exist_ok=True)
        
        env = _env_set(args.env_name, args.env_type)
        env.seed(args.seed) 
        alg = Double(args, env, env_n[0], device, experiment, _dir)
        alg.epsilon_plot()
        alg.train()
    


if __name__ == "__main__":
    _main()
