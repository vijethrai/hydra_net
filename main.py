#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:28:23 2018

@author: raiv
"""

# changes from 1.0
# added noise to input layer
# separate MSE function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import OrderedDict
from datetime import datetime
import datamanager as dm
import runner as runner
from model  import  RNN_Model  
''' 
######################################################################## 
MAIN Function
##########################################################################        
'''

# File and Directory attributes
datadir_gen = '/media/raiv/Data_linux/GDrive_linux/DeepLearning/data/xsens_data/'
config_file='data_config_alpha.csv'
timestamp= datetime.now().strftime("%m_%d_%H_%M")

if __name__ == "__main__":
    


     
  _trainX_Stream,_trainY_Stream,_testX_Stream,_testY_Stream = dm.get_and_reformat_all_data(datadir_gen,config_file)

  print ("Stream _trainX,_trainY,_testX,_testY Shapes", _trainX_Stream.shape,_trainY_Stream.shape,_testX_Stream.shape,_testY_Stream.shape);


 
  param_vals = {'learning_rate': [0.001], 'epoch': [50], 'batch_size':[32,50],
               'seq_length':[5,10,20,100],'num_rnn_layers':[2],'num_hid_units':[4],'reg_parameter':[0.0015,0.0025],'noise_std':[0.01,0.1]}

  model_type='LSTM'
  
  dm.set_global_flags(model_type,timestamp)
  dm.save_settings(param_vals) 

  dm.save_as_mat(['inputs','inputsAndTarget'],dict([ ('trainX_Stream', _trainX_Stream), ('trainY_Stream', _trainY_Stream), ('testX_Stream',_testX_Stream), ('testY_Stream', _testY_Stream) ]))


  runner.train_model(model_type,param_vals,_trainX_Stream,_trainY_Stream,_testX_Stream,_testY_Stream)    
  