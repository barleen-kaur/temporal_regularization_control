import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join

def plot(frame_idx, rewards, losses, log_path):
    
    fig, ax = plt.subplots()
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.xlabel('Number of frames')
    plt.ylabel('Rewards')
    ax.set_axisbelow(True) # Don't allow the axis to be on top of your data
    ax.minorticks_on() # Turn on the minor TICKS, which are required for the minor GRID
    ax.grid(which='major', linestyle='-', linewidth=0.5) # Customize the major grid
    ax.grid(which='minor', linestyle=':', linewidth=0.5) # Customize the minor grid
    ax.tick_params(which='both', # Turn off the display of all ticks.Options for both major and minor ticks
                top=False, # turn off top ticks
                left=False, # turn off left ticks
                right=False,  # turn off right ticks
                bottom=False) # turn off bottom ticks
    plt.savefig(join(log_path,"reward"+".png"))
    plt.close()

    fig, ax = plt.subplots()
    plt.title('Loss with number of frames')
    plt.plot(losses)
    plt.xlabel('Number of frames')
    plt.ylabel('Rewards')
    ax.set_axisbelow(True) # Don't allow the axis to be on top of your data
    ax.minorticks_on() # Turn on the minor TICKS, which are required for the minor GRID
    ax.grid(which='major', linestyle='-', linewidth=0.5) # Customize the major grid
    ax.grid(which='minor', linestyle=':', linewidth=0.5) # Customize the minor grid
    ax.tick_params(which='both', # Turn off the display of all ticks.Options for both major and minor ticks
                top=False, # turn off top ticks
                left=False, # turn off left ticks
                right=False,  # turn off right ticks
                bottom=False) # turn off bottom ticks
    plt.savefig(join(log_path,"loss"+".png"))
    plt.close()


def eps_plot(eps_list, log_path):
    
    fig, ax = plt.subplots()
    plt.plot(eps_list)
    plt.title('Epsilon decay with number of frames')
    plt.xlabel('Number of frames')
    plt.ylabel('Epsilon')
    ax.set_axisbelow(True) # Don't allow the axis to be on top of your data
    ax.minorticks_on() # Turn on the minor TICKS, which are required for the minor GRID
    ax.grid(which='major', linestyle='-', linewidth=0.5) # Customize the major grid
    ax.grid(which='minor', linestyle=':', linewidth=0.5) # Customize the minor grid
    ax.tick_params(which='both', # Turn off the display of all ticks.Options for both major and minor ticks
                top=False, # turn off top ticks
                left=False, # turn off left ticks
                right=False,  # turn off right ticks
                bottom=False) # turn off bottom ticks
    plt.savefig(join(log_path,"eps"+".png"))
    plt.close()
