import os
import json
import random
import argparse
import numpy as np
import sys
from os import path
from os.path import join, basename
from shutil import copy2
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


def _get_cfg():
    parser = argparse.ArgumentParser(description="Main handler for training", usage="./data.sh --algo double --FA deep --env_type atari --game Pong,Breakout --seed 1,2,3,4,5 --beta 0.0,0.1,0.2,0.3,0.7,1.0 --lamb 0.0,0.1,0.2,0.3,0.7,1.0 -lr 1e-3,1e-4,1e-5")
    parser.add_argument('--algo', type=str, default='double', help='algorithm to use: dqn | double')
    parser.add_argument('--seed', type=str, default="1,2,3,4,5", help='random seed (default: 1,2,3,4,5)')
    parser.add_argument('--env_type', type=str, default='atari', help='gym or atari (default: gym)')
    parser.add_argument('--FA', type=str, default='deep', help='linear or deep (default: deep)')
    parser.add_argument('--game', type=str, default='Pong', help='environment to train on (default: Pong)')
    parser.add_argument('--beta', type=str, default='0.0,0.1,0.2,0.3,0.7,1.0', help='beta value (default: 0.0,0.1,0.2,0.3,0.7,1.0)')
    parser.add_argument('--lamb', type=str, default='0.1,0.2,0.3,0.7,1.0', help='lambda value(default: 0.1,0.2,0.3,0.7,1.0)')
    parser.add_argument('-lr', "--lr", type=str, default='1e-3,1e-4,1e-5', help='learning rate (default: 1e-3,1e-4,1e-5)', required=False)
    parser.add_argument('--option', default='copy', help='operation (default: copy)')
    parser.add_argument('--log_dir', default='/scratch/barleenk/temporal/', help='directory to save agent logs (default: /scratch/barleenk/temporal/)')

    args = parser.parse_args()                                            

    return args



###################################################################################


class CopyResults(object):

    def __init__(self, config = "config",
                algo = [],
                seed = [],
                env_type = [],
                FA = [],
                game = [],
                beta = [],
                lamb = [],
                LRs = [],
                log_dir = "afaf"):

        super(CopyResults, self).__init__()
        self.algo = algo
        self.seed = seed
        self.env_type = env_type
        self.FA = FA
        self.game = game
        self.beta = beta
        self.lamb = lamb
        self.LRs = LRs
        self.log_dir = log_dir

    def copy_graphs(self):

        results_dir = join(self.log_dir, "results") 
        os.makedirs(results_dir, exist_ok=True)

        for alg in self.algo:
            for e_type in self.env_type:
                for fa in self.FA:
                    for env in self.game:
                        for s in self.seed:
                            for b in self.beta:
                                if b > 0.0:
                                   for l in self.lamb:
                                       for lr in self.LRs: 
                                           log_file_path = join(self.log_dir, alg+"_results", env, "seed_"+str(s), "beta_"+str(b), "lambda_"+str(l),"lr_"+str(lr),env+"_"+fa+"_training.log")      
                                           result_log_path = join(results_dir, env+"_"+fa+"_"+alg+"_seed"+str(s)+"_beta"+str(b)+"_lamb"+str(l)+"_lr"+str(lr)+"_training.log")
                                           if os.path.isfile(log_file_path):
                                              copy2(log_file_path, result_log_path)

                                           else:
                                              print("{} not found!".format(log_file_path))
                                else:
                                   l = self.lamb[0]
                                   for lr in self.LRs:
                                       log_file_path = join(self.log_dir, alg+"_results", env, "seed_"+str(s), "beta_"+str(b), "lambda_"+str(l),"lr_"+str(lr),env+"_"+fa+"_training.log")  
                                       result_log_path = join(results_dir, env+"_"+fa+"_"+alg+"_seed"+str(s)+"_beta"+str(b)+"_lamb"+str(l)+"_lr"+str(lr)+"_training.log")
                                       if os.path.isfile(log_file_path):                                                                                                                  copy2(log_file_path, result_log_path)
                                       else:                                                                                                                                              print("{} not found!".format(log_file_path))


###################################################################################

def convert(arg_list, type_):
    if type_.find("str") != -1:
        a = [str(item) for item in arg_list.split(',')]
        return a 
    elif type_.find("int") != -1:
        a = [int(item) for item in arg_list.split(',')]
        return a
    elif type_.find("float") != -1:
        a = [float(item) for item in arg_list.split(',')]
        return a

###################################################################################


                                                     
def _main(args):


    algo = convert(args.algo, "str")
    seed = convert(args.seed, "int")
    env_type =  convert(args.env_type, "str")
    FA = convert(args.FA, "str")
    game = convert(args.game, "str")
    beta = convert(args.beta, "float") 
    lamb = convert(args.lamb, "float")
    LRs = convert(args.lr, "float")
    log_dir = args.log_dir


    if args.option == "copy":

        ob = CopyResults(algo = algo,
                        seed = seed,
                        env_type = env_type,
                        FA = FA,
                        game = game,
                        beta = beta,
                        lamb = lamb,
                        LRs = LRs,
                        log_dir = log_dir)
        ob.copy_graphs()



if __name__ == "__main__":
    _main(_get_cfg())
