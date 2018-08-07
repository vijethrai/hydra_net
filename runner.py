#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:04:34 2018

@author: raiv
"""
#runner

import datamanager as dm
from model  import  RNN_Model  
from itertools import product
from collections import defaultdict
from collections import OrderedDict
import time


#main ?
num_trials=5;
bSaveEveryTrial = False;

def append_to_log(log_dict,temp_dict):
   #''' Desc:  appends to loG_dictionary the items in temp dict. Log MUST have
   #          same and all keys
   #     Arguments:  
   #     returns:    appended log dictioneay
   #     @todo :           
   #'''

     for key in temp_dict.keys():         
          #print ("Key",log_dict[key])
          log_dict[key].append(temp_dict[key])                       
     return log_dict

def train_model(model_type,param_vals,_trainX_Stream,_trainY_Stream,_testX_Stream,_testY_Stream)   :
     #sorted by key alphabetical order
     param_vals = OrderedDict(sorted(param_vals.items(), key=lambda t: t[0]))
     model_params=param_vals.fromkeys(param_vals.keys())
     
     #print ("ordererd params", param_vals)
     #stores final results
     log_avg_dict=defaultdict(list)
     log_predictions={"y_Preds":[],"y_Tests":[],"trial_info":[]}
     cnt = 0;
     for p_tups in product(*param_vals.values()):
                        tic = time.clock() 
                        cnt = cnt + 1; 
                        temp_params_dict=dict(zip(param_vals.keys(),p_tups) )
                        model_params.update(temp_params_dict)
                        
     
                         
                        pred_window=0
                        _trainX,_trainY=dm.generateWindows(_trainX_Stream,_trainY_Stream,model_params['seq_length'],pred_window)
                        _testX,_testY=dm.generateWindows(_testX_Stream,_testY_Stream,model_params['seq_length'],pred_window)
                       
                        
                        _trainX,_trainY,_testX,_testY=dm.preProcessData(_trainX,_trainY,_testX,_testY)
 
                        input_shape=_trainX.shape
                        print ("input shape for RNN model",input_shape)
                        nn_model = RNN_Model(model_type,input_shape,model_params)
                               
                               #print some deets
                        #nn_model.printConfig();
                       
                        # averaging of loss for all trials
                        losses_trials_list = [];  
                        least_error = 100 # random hign value to start trials with 
                        
                        
                        for iter_trial in range(num_trials)   :                                                            
                             
                           
                             print ("Training input shapes",(_trainX.shape,_trainY.shape,_testX.shape,_testY.shape))
                             this_trial_predictions,this_trial_error=nn_model.train_data(_trainX,_trainY,_testX,_testY)
                                                                                                                                                         
                             losses_trials_list.append(this_trial_error);
                                       
                             print ('\n  Trial # %d Test MSE %f'%(iter_trial,this_trial_error))
                             
                             # save every trials predictions 
                             if bSaveEveryTrial:
                               print ("Saving every trial results, Trial #:",iter_trial)   
                               log_predictions['y_Preds'].append(this_trial_predictions); log_predictions['y_Tests'].append(_testY) ; log_predictions['trial_info'].append(model_params)
                             elif this_trial_error < least_error:
                               print ("Saving this trial with error %f less than previous best of %f"%(this_trial_error,least_error)) 
                               log_predictions['y_Preds'] = this_trial_predictions; log_predictions['y_Tests'] = _testY ; log_predictions['trial_info'] = model_params
                               least_error=this_trial_error;

                        toc = time.clock()
                        print ("time elapsed for this parameter set",toc - tic)
                        # save trials predictions,  one file for  every unique hyperparameter setting
                        dm.save_as_mat(['hyperparams','predictions' + str(cnt)],log_predictions) 
                        
                        #avg loss for these parameters for all trials
                        loss_out_avg =  float(sum(losses_trials_list)/len(losses_trials_list));
                        print ('number of trials',len(losses_trials_list))
                               
                        # save avg loss for these parameters
                        temp_params_dict['param_idx']= cnt; temp_params_dict['error']= loss_out_avg
                        
                  
                        log_avg_dict = append_to_log(log_avg_dict,temp_params_dict) 
                        
                        
                        
                        print ("\n model params",temp_params_dict.items())  
                        print ("\n Avg loss out: %f, this param set # %d"%(loss_out_avg,cnt))            
         
                        
     
       
     #  save average performance all parameter combinations in one file      
     dm.save_as_mat(['hyperparams','average'],log_avg_dict) 
      


     #  plt.plot(pred_out_this_trial, 'r-', _testY, 'g-')
     ##