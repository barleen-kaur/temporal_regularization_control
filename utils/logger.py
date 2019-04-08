
import numpy as np
import pandas as pd
import os
from os.path import join



class Logger(object):

    def __init__(self, mylog_path="./log", mylog_name="training.log", mymetric_names=["rewards"] ):
        super(Logger, self).__init__()
        self.log_path = mylog_path
        self.log_name = mylog_name
        self.metric_names = list(mymetric_names)

    def to_csv(self, metric_array, nb_episode):
        
        if nb_episode == 1:
            met_c =  self.metric_names
            df = pd.DataFrame(columns=met_c)
            df.loc[0] = metric_array
        else:
            df = pd.read_csv(join(self.log_path,self.log_name), index_col=0)
            df.loc[len(df)] = metric_array
        df.to_csv(join(self.log_path,self.log_name))
