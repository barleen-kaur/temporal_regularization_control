import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join


class LossPlotter(object):

    def __init__(self, mylog_path="./log", mylog_name="training.log", env_name = "CartPole", xmetric_name = 'frame', ymetric_names=[ 'episode_return', 'loss']):
        super(LossPlotter, self).__init__()
        self.log_path = mylog_path
        self.log_name = mylog_name
        self.env_name = env_name
        self.xmetric_name = xmetric_name
        self.metric_names = list(ymetric_names)
        os.makedirs(join(self.log_path, "plot"), exist_ok=True)

    
    def plotter(self):

        dataframe = pd.read_csv(join(self.log_path,self.log_name), skipinitialspace=True)

        for i in range(len(self.metric_names)):
            fig, ax = plt.subplots()
            plt.plot(list(dataframe[self.xmetric_name]),list(dataframe[self.metric_names[i]]),label=self.metric_names[i])
            plt.xlabel('Number of frames')
            plt.ylabel(self.metric_names[i])
            ax.set_axisbelow(True) # Don't allow the axis to be on top of your data
            ax.minorticks_on() # Turn on the minor TICKS, which are required for the minor GRID
            ax.grid(which='major', linestyle='-', linewidth=0.5) # Customize the major grid
            ax.grid(which='minor', linestyle=':', linewidth=0.5) # Customize the minor grid
            ax.tick_params(which='both', # Turn off the display of all ticks.Options for both major and minor ticks
                        top=False, # turn off top ticks
                        left=False, # turn off left ticks
                        right=False,  # turn off right ticks
                        bottom=False) # turn off bottom ticks
            plt.savefig(join(self.log_path,"plot",self.env_name+"_"+self.metric_names[i]+".png"))
            plt.close()
   


def eps_plot(eps_list, log_path, env_name):
    
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
    plt.savefig(join(log_path,"plot", env_name+"_eps.png"))
    plt.close()
