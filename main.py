

import os
import time
import numpy as np
from collections import deque
from comet_ml import OfflineExperiment, Experiment

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


args = get_args()

path = ""

assert args.algo in ['dqn', 'double']

if args.comet == "offline":
    experiment = OfflineExperiment(project_name="recurrent-value", workspace="nishanthvanand",disabled=args.disable_log, offline_directory="../comet_offline", parse_args=False)
elif args.comet == "online":
    experiment = Experiment(api_key="piz8lnPU3GY0GPZQ10mef0rdU", project_name="temporal-reg-control", workspace="barleenk", disabled=args.disable_log, parse_args=False)
else:
    raise ValueError

experiment.log_parameters(vars(args))

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes #change this

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#figure out directory stuff
try:
    os.makedirs(os.path.join(path, args.log_dir), exist_ok=True) #change this
    eval_log_dir = args.log_dir + "_eval"
    os.makedirs(eval_log_dir)



#########################################################################################################

def _env_set(env_name, env_type):

    if env_type == 'gym':
            env = gym.make(env_name)

    elif env_type == 'atari':
        env = make_atari(env_name)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)

    return env

def _main():
    
    torch.set_num_threads(1). #what's this?
    device = torch.device("cuda:0" if args.cuda else "cpu")


    if args.algo == 'dqn':
        agent = algo.DQN(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               lr_beta=args.lr_beta, reg_beta=args.reg_beta,
                               delib_center=args.delib_center,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)

    elif args.algo == 'double':
        _env = _env_set(args.env_name, arg.env_type)
        agent = algo.DOUBLE(args, _env, device)
    


if __name__ == "__main__":
    main()