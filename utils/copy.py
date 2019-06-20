import os
import json
import random
import argparse
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


def _get_cfg():
    parser = argparse.ArgumentParser(description="Main handler for training", usage="./data.sh --algo double --seed 1 --env_type atari --FA deep --env_name Pong --beta 0.1 --lamb 0.1")
    parser.add_argument('--algo', default='double', help='algorithm to use: dqn | double')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--env_type', default='gym', help='gym or atari (default: gym)')
    parser.add_argument('--FA', default='deep', help='linear or deep (default: deep)')
    parser.add_argument('--env_name', default='Pong', help='environment to train on (default: Pong)')
    parser.add_argument('--beta', type=float, default=0.0, help='beta value (default: 0.0)')
    parser.add_argument('--lamb', type=float, default=0.1, help='lambda value(default: 0.1)')
    parser.add_argument('--log_dir', default='/scratch/barleenk/temporal/', help='directory to save agent logs (default: /tmp/gym)')

    args = parser.parse_args()                                            

    return args


args = _get_cfg()


###################################################################################


class CopyResults(object):

    def __init__(self, config = "config",
                algo = [],
                seed = [],
                env_type = [],
                FA = [],
                env_name = [],
                beta = [],
                lamb = [],
                log_dir = "\vafaf"):

        super(CopyResults, self).__init__()
        self.algo = algo
        self.seed = seed
        self.env_type = env_type
        self.FA = FA
        self.env_name = env_name
        self.beta = beta
        self.lamb = lamb
        self.log_dir = log_dir
        #self.tumor = ["Enhance", "Whole", "Core", "class_0", "class_1", "class_2"]
        #self.tumor_class = ["en", "wh", "co", "0", "1", "2"]
        #self.tags = ["Baseline" , "Finetune Decoder", "Finetune Last three", "Finetune All"]
        #self.thresholds = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999]
        #self.roc_metrics = roc_metrics

    def copy_graphs(self):

        results_dir = join(self.log_dir, "results") 
        os.makedirs(results_dir, exist_ok=True)

        for alg in self.algo:
            for s in self.seed:   
                for e_type in self.env_type:
                    for fa in self.FA:
                        for env in self.env_name:
                            for b in self.beta:
                                for l in self.lamb:
                                    
                                    log_file = join(self.log_dir, alg+"_results", env, "seed_"+str(s), "beta_"+str(b), "lambda_"+str(l)) 
                                    log_name = env+"_"+fa+"_training.log" 

                                    log_path = join(log_file, log_name)
                                    exists = os.path.isfile(log_path)
                                    results_log_name = env+"_"+fa"_"+alg"_seed"+str(s)"_beta"+str(b)"_lamb"+str(l)+"_training.log"
                                    result_log_path = join(results_dir, results_log_name)
                                    if exists:
                                        copy2(log_path, result_log_path)

                                    else:
                                        print("{} not found!".format(log_path))




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
    env_name = convert(args.env_name, "str")
    beta = convert(args.beta, "float")                                                
    lamb = convert(args.lamb, "float")
    log_dir = args.log_dir

    if args.option == "copy":

        ob = CopyResults(algo = algo,
                        seed = seed,
                        env_type = env_type,
                        FA = FA,
                        env_name = env_name,
                        beta = beta,
                        lamb = lamb,
                        log_dir = log_dir)
        ob.copy_graphs()



if __name__ == "__main__":
    _main(_get_cfg())