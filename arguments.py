import argparse
import torch


def get_args():
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='double', help='algorithm to use: dqn | double')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer name (default: adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--eps_s', type=float, default=1.0, help='epsilon-greedy start value (default: 1.0)')
    parser.add_argument('--eps_f', type=float, default=0.01, help='epsilon-greedy final value (default: 0.01)')
    parser.add_argument('--eps_decay', type=float, default=500, help='epsilon-greedy decay rate (default: 500)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size(default: 32)')
    parser.add_argument('--replay_buff', type=int, default=1000, help='size of replay buffer (default: 1000)')
    parser.add_argument('--plot_idx', type=int, default=500, help='frequency with which to plot the graph')
    parser.add_argument('--target_idx', type=int, default=100, help='no of steps for updating target network')
    parser.add_argument('--checkpoint_idx', type=int, default=50000, help='checkpointing frequency in terms of no of episodes')
    parser.add_argument('--env_type', default='gym', help='gym or atari (default: gym)')
    parser.add_argument('--FA', default='deep', help='linear or deep (default: deep)')
    parser.add_argument('--env_name', default='CartPole-v0', help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--num_frames', type=int, default=250000, help='number of frames (default: 250000)')
    parser.add_argument('--start_frame', type=int, default=0, help='frame number to start training (default: 0)')
    parser.add_argument('--frame_stack', type=bool, default=False, help='frame stack : True/False (default: False)')
    parser.add_argument('--frame_skip', type=int, default=4, help='number of frame to skip (default: 4)')
    parser.add_argument('--beta', type=float, default=0, help='beta value (default: 0)')
    parser.add_argument('--lamb', type=float, default=0.1, help='lambda value(default: 0.1)')
    parser.add_argument('--return_deque', type=int, default=20, help='size of deque for storing return in previous n recent episodes(default: 20)')
    parser.add_argument('--comet', type=str, default="online")
    parser.add_argument('--disable_log', type=bool, default=True, help='set it to True to disable comet logging')
    parser.add_argument('--log_dir', default='/scratch/barleenk/temporal/', help='directory to save agent logs (default: /tmp/gym)')
 
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

