#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:59:04 2018

@author: raiv
"""

""" 
============================================================================================================
These are functions for Data management 
-loading
-processing
-formatting
-saving
============================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import sleep
import math
import os
import scipy.io as sio 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from skimage.util.shape import view_as_windows
from math import sqrt
import pandas as pd
import itertools

  


b_PrintShapes=True

results_dir =[]
timestamp =[]
model_type=[]
b_normalize_data=False;
b_standardize_data=False;
l_norm_scaler =[[],[]]; #scalers for norm
data_config_df=pd.DataFrame()

   #''' Desc: 
   #     Arguments: 
   #     returns:    
   #     @todo :           
   #'''

'''
# Data Loading and Saving functions    
'''



def get_and_reformat_all_data(datadir_gen,config_filename):
     
    global results_dir  
    global data_config_df
    bTarget_separate_dir=False; #by default Input and Target features are in same directories
    bTest_separate_dir = True; # By default test and train are different directories
    exclude_str_list = []
    
    def concatenate_all_values_with(datadir_gen,values_list):          
         return [datadir_gen + value for value in values_list]

    config_file = datadir_gen + config_filename
    data_config_df=pd.read_csv(config_file,index_col=0,header=0,engine='python')

    #Train directory and features 
    train_datadir_list= concatenate_all_values_with(datadir_gen, data_config_df.loc['train_dir'].dropna().tolist() )
    
    include_str=data_config_df.loc['train_include_features'].dropna().values[0]
    #print ('include_str',include_str)
    
    
    if not(data_config_df.loc['train_exclude_features'].dropna().empty):
             
             for i in range( data_config_df.loc['train_exclude_features'].count()):
                  if type(data_config_df.loc['train_exclude_features'].values[i])==str:
                     print (type(data_config_df.loc['train_exclude_features'].values[i]))
                     exclude_str_list.append(data_config_df.loc['train_exclude_features'].values[i])
#                     continue;
                  else:
                      break;
#                 if type(data_config_df.loc['train_exclude_features'].values[i])==float:
#                     print (type(data_config_df.loc['train_exclude_features'].values[i]))
#                     continue;
#                 else:
#                     print (type(data_config_df.loc['train_exclude_features'].values[i]))
#                     exclude_str_list.append(data_config_df.loc['train_exclude_features'].values[i])
             
   # print ('exclude_str',exclude_str)

               
         
    
    #Target directory and feature
    #empty Target dir indicates, target labels are in the same folder as input features
    if not(data_config_df.loc['target_dir'].dropna().empty) :
         # Target files not in training features folders
         bTarget_separate_dir=True;
         
         target_datadir_list= concatenate_all_values_with(datadir_gen, data_config_df.loc['target_dir'].dropna().tolist() )
         y_filename_list= data_config_df.loc['target_feature'].dropna().tolist()
    else:
         print ("Targets in same folder as inputs")
         target_datadir_list=train_datadir_list;
         y_filename_list= data_config_df.loc['target_feature'].dropna().tolist()

    
    #Test set dir 
    #empty test_dir field value indicates Test will be split from training data, hence same folder
    if data_config_df.loc['test_dir'].dropna().empty :
        
         test_train_split=int(data_config_df.loc['test_train_split'].values[0])
         print ("test_train_split",test_train_split)
         test_datadir_list = train_datadir_list;
         bTest_separate_dir=False;
    else:  
         # test is different directory and needs to be loaded separately
         test_datadir_list=concatenate_all_values_with(datadir_gen, data_config_df.loc['test_dir'].dropna().tolist()) 
    
    
     #Test set dir 
    #empty test_dir field value indicates Test will be split from training data, hence same folder
    if not( data_config_df.loc['val_dir'].dropna().empty) :
         # test is different directory and needs to be loaded separately
         val_datadir_list=concatenate_all_values_with(datadir_gen, data_config_df.loc['val_dir'].dropna().tolist()) 
    
    #results
    results_dir=concatenate_all_values_with(datadir_gen, data_config_df.loc['results_dir'].dropna().tolist() )
    
    #Normalize or not
    
    
    
    
    print ("Loading data configuration file from folder:",datadir_gen)
    print (" training directories:",train_datadir_list)
    if bTarget_separate_dir:
         print ('Target located in folder',target_datadir_list)
    print ('Target feature name',y_filename_list[0])     
    print ("Test directory",*test_datadir_list)
    print ("Results will be stored in",results_dir)

#    
#
#    
#
    #one of the data directories is sufficient to get filenames
    
    input_x_filenames_list = get_input_feature_filenames(train_datadir_list,include_str,exclude_str_list)
  


    if  bTest_separate_dir:
         #load test and train set separately
         _trainX_Stream,_trainY_Stream=load_files_from_folders(train_datadir_list,input_x_filenames_list,target_datadir_list,y_filename_list)
         _testX_Stream,_testY_Stream=load_files_from_folders(test_datadir_list,input_x_filenames_list,test_datadir_list,y_filename_list)
         _valX_Stream,_valY_Stream=load_files_from_folders(val_datadir_list,input_x_filenames_list,val_datadir_list,y_filename_list)

    else: 
         #load just from train directory once and split 
        _trainX_Stream,_trainY_Stream=load_files_from_folders(train_datadir_list,input_x_filenames_list,target_datadir_list,y_filename_list)
        _trainX_Stream,_trainY_Stream,_testX_Stream,_testY_Stream=splitTestTrainData(_trainX_Stream,_trainY_Stream,test_train_split)
        
         
     
    return _trainX_Stream,_trainY_Stream,_testX_Stream,_testY_Stream,_valX_Stream,_valY_Stream

def get_input_feature_filenames(datadir, include_str, exclude_str_list):
    ''' Basic function to get filenames of input features in a folder
        Arguments: data_dir: one of the data directories is sufficient to get filenames
                   include_str: include files with this string
                   exclude_str: exclude files with this string
        returns: list of all filenames to be loaded as inputs           
        @todo include feature not active now           
    '''

  #x_feature_types=[];
    x_feature_all_list=os.listdir(datadir[0]);
    
    if  not(include_str == 'all'):   
        print ("Including only",include_str)
        x_feature_all_list = [s for s in x_feature_all_list if include_str in str(s)]
    
    ex_matches=[]
    if not(exclude_str_list[0] =='None'):
         print ("exclude_str_list",len(exclude_str_list))
         for idx_ex in range(len(exclude_str_list)):
             exclude_str=exclude_str_list[idx_ex]
             
             this_ex_matches=[s for s in x_feature_all_list if exclude_str in str(s)]
            
             ex_matches= itertools.chain(ex_matches,this_ex_matches)
             print( ex_matches)
             
             
         for match in ex_matches :
             print ("Excluding: ",match) 
             x_feature_all_list.pop(x_feature_all_list.index(match))
        
        
    return x_feature_all_list;    



def load_files_from_folders(x_dir_list,x_feature_names_list,y_dir_list,y_feature_name_list):
    ''' Desc: loads all files with same name from multiple folders
        Arguments: list of directory names to load from
                   x_feature_name: names of input features
                   y_feature_name: name of target feature 
        returns:   input and output of shape (samples, seq_length, input_features) 
        @todo :           
    '''
    
    
    print ("")
   
    b_Init=True;
    thisFeatureData_X=[];
    for x_curr_feature in x_feature_names_list :
      
     thisFeatureData_X=np.array([])
     # load train data
     print ("Xfeatures",x_curr_feature)
     for datadir in x_dir_list:
         
         thisFeatureData_X_part=np.loadtxt(datadir+  x_curr_feature,delimiter=',')                 
         thisFeatureData_X=np.hstack((thisFeatureData_X,thisFeatureData_X_part))
         
     if b_Init==True:
         multiFeatures_X=thisFeatureData_X;         
         b_Init=False;
     else:    
         multiFeatures_X=np.dstack((multiFeatures_X,thisFeatureData_X))
          
   
    # load feature set that will be target
    print ("Target Feature",y_feature_name_list[0])
    y_=np.array([])
    
    for datadir in y_dir_list:
      
       y_part=np.loadtxt(datadir+  y_feature_name_list[0],delimiter=',')   
       y_=np.hstack((y_,y_part))
       
    x_=multiFeatures_X
    print (" Data loaded\n")
   
    if b_PrintShapes: print ('Initial Load Data x.shape, y.shape',x_.shape, y_.shape)
    return x_, y_

def splitTestTrainData(x_,y_,testset_percent):
   n_examples,n_timesteps,n_features = x_.shape
   
   if n_examples ==1 :
        #split in time axis
        split_axis=1;       
        train_len = n_timesteps- int(n_timesteps/testset_percent)   
        
        
   _trainX=[]; _trainY=[]; _testX=[]; _testY=[]
  

   #y_=np.reshape(y_,(-1,1))
       
   _trainX,_testX= np.split(x_,[train_len],axis=split_axis);
   _trainY,_testY=np.split(y_,[train_len])
   
   
   if b_PrintShapes: print ('Post Split Train X, test X shapes',_trainX.shape,_testX.shape)
   if b_PrintShapes: print ('Post Split Train Y, test Y shape',_trainY.shape,_testY.shape)
   
   return _trainX,_trainY,_testX,_testY;

   

   
def getUniqueFileName(scope,var_name):
     
    global timestamp
    global model_type
    
       
#    print ("type resultsdir",type(results_dir[0])) 
#    print ("type timestamp",type(timestamp)) 
#    print ("type model_type",type(model_type)) 
    
 
     
    timstamped_filename = results_dir[0] + scope + "/" 
    
    
    if not os.path.exists(timstamped_filename):
         os.makedirs(timstamped_filename)
    
    
    timstamped_filename=timstamped_filename+ var_name  
    
    #print ("timstamped_filename",timstamped_filename)
    
    return str(timstamped_filename);

def get_python_results_dir():
     
     results_dir + model_type  + timestamp + '/python/'
     
def get_results_dir():     
    return results_dir[0]     

def set_results_dir():
    
     results_dir[0] = results_dir[0] + model_type +'/'+ timestamp+'/' 
     
def make_folder_in_results(folder_name) :    

    if not os.path.exists(folder_name):
         os.makedirs(folder_name)
    
#def set_results_dir(this_results_dir):
#     global results_dir;
#     
#     results_dir=this_results_dir;
#     return;

def save_as_mat(matfilename_str_list,vars_dict):
    ''' Desc: saves variables as matfile
       Arguments: filename_str [scope (i.e foldername),variable name] 
                 vars_dict:dictionary of variables
       returns:   nada 
       @todo :  kiss your mom         
   '''
    
    scope    =   matfilename_str_list[0];
    var_name =  matfilename_str_list[1];
    

    print('Saving %s in %s folder: '%(var_name,scope))

    matlabFileName = getUniqueFileName(scope,var_name)
     
    sio.savemat(matlabFileName,vars_dict)
    
    return
def set_global_flags(m_type,tstamp):
        #''' Desc: sets the global flags needed to save various results once and for all from the main file
        #     Arguments: model_type,timestamp
        #     returns:    
        #     @todo :      maybe make it dictionary     
        #'''

     
     global timestamp
     global model_type
     global results
     
     timestamp=tstamp
     model_type = m_type
     
     print ("Datamanager flags updated times stamp and model tpe", timestamp,model_type)
     
     set_results_dir()
     print ("Results for this run will be stored in",results_dir[0])
     return 

def save_settings(param_vals)     :
     #data_config_df=pd.read_csv(config_file,index_col=0,header=0,engine='python')

#     p_df=pd.DataFrame.from_dict(param_vals)
     global data_config_df
     p_df=pd.DataFrame.from_dict(dict([ (k,pd.Series(v)) for k,v in param_vals.items() ]),orient='index')
     data_config_df=data_config_df.append(p_df)
     
     timstamped_path=results_dir[0] 
     if not os.path.exists(timstamped_path):
         os.makedirs(timstamped_path)
         
     data_config_df.to_csv(path_or_buf=timstamped_path +'run_settings.csv ')
     
     return 
''''
#############################################################################
data preprocessing functions     
##############################################################################
''' 
def generateWindows(x_,y_,sliding_window,pred_window):
    
    # Input: multi dimensional continuous x and 1D y
    # output multi dim x w/ each row as sliding window 
    #        list of targets y for different prediction windows
    
    
        #prep windows for sliding window
        
    #pred_window_max= max(l_d_pred_window)
    window_shape_x = (sliding_window,)
    window_shape_y=(sliding_window+pred_window,)
    
    b_Init= True; 
    n_examples,n_timesteps,n_features = x_.shape
    
    #check n_examples is just one
    if n_examples != 1:
        print ('Not a 1 Dimensional continuous stream input')

    
    
    if b_PrintShapes: print ("Gen Windows input x_.shape",x_.shape)
    if b_PrintShapes: print ("Gen Windows input y_.shape",y_.shape)

    for idx_feature in range(n_features):
    # remove last values of x to leave room for predicted values in y
        this_feature_x = x_[0,:,idx_feature]
        
        if pred_window ==0:
            
            this_feature_x=np.transpose(this_feature_x)
            
        else: # remove values with no prediction possibilities
            
            this_feature_x=np.transpose(this_feature_x[:-pred_window])
        
        #print ('this X feature shape',this_feature_x.shape)

        x_win = view_as_windows(this_feature_x,window_shape_x)
    
        #print ('X_win',x_win.shape)
        if b_Init==True:
         multiFeatures_X_win=x_win;
         b_Init=False;
        else:    
         multiFeatures_X_win=np.dstack((multiFeatures_X_win,x_win))
    
    
    y_win = view_as_windows(y_, window_shape_y)  


    if b_PrintShapes: print ('Gen Win Output X_win shape',multiFeatures_X_win.shape)
    if b_PrintShapes: print ('Gen Win Output Y_win shape',y_win.shape)
    
    # return the single value corresponding to prediction window length
    y_SingleTarget=y_win[:,sliding_window+pred_window-1]

    return multiFeatures_X_win,y_SingleTarget






def normalizeEachFeature(list_unNormalizedData, b_Train):   
   
   # this is for sliding window edition
   # 
#    print (" ")
#    print ("Normalization Function")
    
    global l_norm_scaler;
    b_singleFeature=False;
    
    list_normalized=[]
    for idx_dataset in range(len(list_unNormalizedData)):
        
        dataset =      list_unNormalizedData[idx_dataset]  
        if b_PrintShapes: print (" unnormalized data shape", dataset.shape )        
        
        if len(dataset.shape)==3:
            nsamples, n_tsteps, n_features = dataset.shape
            
        elif len(dataset.shape)==2:
            # column vector of single valued targets only
            
            nsamples = dataset.shape
            n_features=1;nsamples=1;
           
            #dataset=dataset.reshape((nsamples,n_features))
            b_singleFeature=True;
        
        
        #print (nsamples, n_tsteps, n_features)
        normalized_dataset=np.empty_like(dataset)

        for iter_feature in range(n_features):
            
            if b_singleFeature == False:
                this_feature=dataset[...,iter_feature]
            else:
                this_feature=dataset
                
            #reshape this slice into column vect
            #this_feature_1D=np.reshape(this_feature,(-1,1))
            this_feature_1D=   this_feature
            if b_PrintShapes : print ("This feature shape",this_feature_1D.shape) 
            # fit scaler for training data only and save in global list of norm scalers for 
            # each featurei
            if b_Train == True:
                norm_scaler_feature = MinMaxScaler(feature_range=(0, 1))
                norm_scaler_feature = norm_scaler_feature.fit(this_feature_1D)
                print ("Min Max Scale",norm_scaler_feature.data_min_,norm_scaler_feature.data_max_,norm_scaler_feature.scale_)
                l_norm_scaler[idx_dataset].append(norm_scaler_feature)
            else:
                #if test set, use scalers fitted to corresponding train set feature
                norm_scaler_feature=l_norm_scaler[idx_dataset][iter_feature]
            
            #print('data_idx:%f, feature_idx: %f Min: %f, Max: %f' % (idx_dataset, iter_feature,norm_scaler_feature.data_min_, norm_scaler_feature.data_max_))
            # normalize the dataset and print
            normalized_feature = norm_scaler_feature.transform(this_feature_1D)
            
            
            if b_singleFeature ==False:
                normalized_feature=normalized_feature.reshape((nsamples, n_tsteps))
                normalized_dataset[...,iter_feature]=normalized_feature;
                
            else:
                
                normalized_dataset=normalized_feature;
                #b_singleFeature=False;
    
        

        # inverse transform and print
        #inversed = scaler.inverse_transform(normalized)
        #print(inversed)
        if b_PrintShapes: print ("Normalized dataset shape",normalized_dataset.shape) 
        list_normalized.append(normalized_dataset)
        
    return list_normalized

def clearNormScalerList():
     global l_norm_scaler
     l_norm_scaler=[[],[]]
     
     return;
     
def standardizeEachFeature(list_nonStdData,b_Train):   
   
   # this is for sliding window edition
   # 
#    print (" ")
    print ("Gaussian Standardize  Function")
    
    b_singleFeature=False;
    
    list_std=[]
    for idx_dataset in range(len(list_nonStdData)):
        
        dataset =  list_nonStdData[idx_dataset]        
        if len(dataset.shape)==3:
            nsamples, n_tsteps, n_features = dataset.shape
            
        
        elif len(dataset.shape)==1:
            # column vector of single valued targets only
            
            nsamples = dataset.shape
            n_features=1;n_tsteps=1;
           
            #dataset=dataset.reshape((nsamples,n_features))
            b_singleFeature=True;
        
        
        #print (nsamples, n_tsteps, n_features)
        std_dataset=np.empty_like(dataset)

        for iter_feature in range(n_features):
            
            if b_singleFeature == False:
                this_feature=dataset[...,iter_feature]
            else:
                this_feature=dataset
                
            #reshape this slice into column vect
            this_feature_1D=np.reshape(this_feature,(-1,1))

            
            if b_Train == True:
                 std_scaler_feature = StandardScaler()
                 std_scaler_feature = std_scaler_feature.fit(this_feature_1D)
                 l_std_scaler[idx_dataset].append(std_scaler_feature)
            else:
                #if test set, use scalers fitted to corresponding train set feature
                std_scaler_feature=l_std_scaler[idx_dataset][iter_feature]
                         
           # print('data_idx:%f, feature_idx: %f Mean: %f, StandardDeviation: %f' % (idx_dataset, iter_feature,std_scaler_feature.mean_, sqrt(std_scaler_feature.var_)))            # normalize the dataset and print
            std_feature = std_scaler_feature.transform(this_feature_1D)
            
            
            if b_singleFeature ==False:
                std_feature=std_feature.reshape((nsamples, n_tsteps))
                std_dataset[...,iter_feature]=std_feature;
                
            else:
                
                std_dataset=std_feature;
                #b_singleFeature=False;
    
        

        # inverse transform and print
        #inversed = scaler.inverse_transform(normalized)
        #print(inversed)
        list_std.append(std_dataset)
        
    print (" ")
    return list_std



def preProcessData(_trainX,_trainY,_testX,_testY,_valX,_valY):

   if b_PrintShapes: print ('Processed Data Input Train X, test X shapes,Val X shape',_trainX.shape,_testX.shape,_valX.shape)
   if b_PrintShapes: print ('Processed Data Input Train Y, test Y shape,Val Y shape',_trainY.shape,_testY.shape,_valY.shape)
       
   # reshape targeet from [batch_size,] to [batch_size,1]
      
   _trainY=np.reshape(_trainY,(-1,1))
   _testY=np.reshape(_testY,(-1,1))
   _valY=np.reshape(_valY,(-1,1))
   
   if b_normalize_data == True:
       
       print ("saving normalized")
       #save_as_mat(['inputs','unNormalized'],dict([ ('u_trainX', _trainX), ('u_trainY', _trainY),('u_testX',_testX),('u_testY',_testY),('u_valX',_valX),('u_testY',_valY)]))

       
       [_trainX,_trainY]=normalizeEachFeature([_trainX,_trainY],b_Train=True)
       [_testX,_testY]=normalizeEachFeature([_testX,_testY],b_Train=False)
       [_valX,_valY]=normalizeEachFeature([_valX,_valY],b_Train=False)
       #save_as_mat(['inputs','Normalized'],dict([ ('n_trainX', _trainX), ('n_trainY', _trainY),('n_testX',_testX),('n_testY',_testY)]))

       #clear scaler after train and test have been normalized
       
       clearNormScalerList()
       
   elif b_standardize_data == True :
       
       [_trainX,_trainY]=standardizeEachFeature([_trainX,_trainY],b_Train=True)
       [_testX,_testY]=standardizeEachFeature([_testX,_testY],b_Train=False)
             
  
      

   
   
   if b_PrintShapes: print ('Processed Data o/p Train X, test X shapes',_trainX.shape,_testX.shape,_valX.shape)
   if b_PrintShapes: print ('Processed Data o/p Train Y, test Y shape',_trainY.shape,_testY.shape,_valY.shape)
   
   return _trainX,_trainY,_testX,_testY,_valX,_valY

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    
    return a,b
    
    
    '''
        Junk Deprecated etc
    '''
##################################3
    #def get_and_reformat_all_data(datadir_gen,config_filename):
#     
#    global results_dir  
#    
#    def concatenate_all_values_with(datadir_gen,values_list):          
#         return [datadir_gen + value for value in values_list]
#
#    config_file = datadir_gen + config_filename
#    data_config_df=pd.read_csv(config_file,index_col=0,header=0,engine='python')
#
#     
#    train_datadir_list= concatenate_all_values_with(datadir_gen, data_config_df.loc['train_dir'].dropna().tolist() )
#    
#    
#    test_datadir=concatenate_all_values_with(datadir_gen, data_config_df.loc['test_dir'].dropna().tolist()) 
#    results_dir=concatenate_all_values_with(datadir_gen, data_config_df.loc['results_dir'].dropna().tolist() )
#    
#    
#    
#    
#    print ("Loading data configuration file from folder:",datadir_gen)
#    print (" training directories:",train_datadir_list)
#    print (" Test directory",test_datadir)
#    print ("Results will be stored in",results_dir)
#
#    y_filename_list=['a_jLeftAnkle_sag_stream'] ;  
#
#    include_str=['all']; exclude_str='LeftAnkle'  
#
#    #one of the data directories is sufficient to get filenames
#    
#    input_x_filenames_list = get_input_feature_filenames(train_datadir_list,include_str,exclude_str)
#  
#
#    _trainX_Stream,_trainY_Stream=load_files_from_folders(train_datadir_list,input_x_filenames_list,y_filename_list)
#    _testX_Stream,_testY_Stream=load_files_from_folders(test_datadir,input_x_filenames_list,y_filename_list)
#
#     
#    return _trainX_Stream,_trainY_Stream,_testX_Stream,_testY_Stream
#
#

#
#def get_input_feature_filenames(datadir, include_str, exclude_str):
#    ''' Basic function to get filenames of input features in a folder
#        Arguments: data_dir: one of the data directories is sufficient to get filenames
#                   include_str: include files with this string
#                   exclude_str: exclude files with this string
#        returns: list of all filenames to be loaded as inputs           
#        @todo include feature not active now           
#    '''
#
#  #x_feature_types=[];
#    x_feature_all_list=os.listdir(datadir[0]);
#    
#
#    ex_matches = [s for s in x_feature_all_list if exclude_str in str(s)]
#    
#    
#    for match in ex_matches :
#        
#        x_feature_all_list.pop(x_feature_all_list.index(match))
#        
#        
#    return x_feature_all_list;    
#
#
#
#def load_files_from_folders(dir_list,x_feature_names_list,y_feature_name_list):
#    ''' Desc: loads all files with same name from multiple folders
#        Arguments: list of directory names to load from
#                   x_feature_name: names of input features
#                   y_feature_name: name of target feature 
#        returns:   input and output of shape (samples, seq_length, input_features) 
#        @todo :           
#    '''
#    
#    
#    print ("")
#   
#    b_Init=True;
#    thisFeatureData_X=[];
#    for x_curr_feature in x_feature_names_list :
#      
#     thisFeatureData_X=np.array([])
#     # load train data
#     print ("Xfeatures",x_curr_feature)
#     for datadir in dir_list:
#         
#         thisFeatureData_X_part=np.loadtxt(datadir+  x_curr_feature,delimiter=',')                 
#         thisFeatureData_X=np.hstack((thisFeatureData_X,thisFeatureData_X_part))
#         
#     if b_Init==True:
#         multiFeatures_X=thisFeatureData_X;         
#         b_Init=False;
#     else:    
#         multiFeatures_X=np.dstack((multiFeatures_X,thisFeatureData_X))
#          
#   
#    # load feature set that will be target
#    print ("Target Feature",y_feature_name_list[0])
#    y_=np.array([])
#    for datadir in dir_list:
#      
#       y_part=np.loadtxt(datadir+  y_feature_name_list[0],delimiter=',')   
#       y_=np.hstack((y_,y_part))
#       
#    x_=multiFeatures_X
#    print (" Data loaded\n")
#   
#   
#    return x_, y_

#def generateWindows_legacy(x_,y_,sliding_window,l_d_pred_window):
#    
#    # Input: multi dimensional continuous x and 1D y
#    # output multi dim x w/ each row as sliding window 
#    #        list of targets y for different prediction windows
#    
#    
#        #prep windows for sliding window
#        
#    pred_window_max= max(l_d_pred_window)
#    window_shape_x = (sliding_window,)
#    window_shape_y=(sliding_window+pred_window_max,)
#    
#    b_Init= True; 
#    n_examples,n_timesteps,n_features = x_.shape
#    
#    #check n_examples is just one
#    if n_examples != 1:
#        print ('Not a 1 Dimensional continuous stream input')
#
#    
#    
#    print ("Gen Windows input x_.shape",x_.shape)
#    print ("Gen Windows input y_.shape",y_.shape)
#
#    for idx_feature in range(n_features):
#    # remove last values of x to leave room for predicted values in y
#        this_feature_x = x_[0,:,idx_feature]
#        
#        if pred_window_max ==0:
#            
#            this_feature_x=np.transpose(this_feature_x)
#            
#        else: # remove values with no prediction possibilities
#            
#            this_feature_x=np.transpose(this_feature_x[:-pred_window_max])
#        
#        
#            
#            
#        
#        #print ('this X feature shape',this_feature_x.shape)
#
#        x_win = view_as_windows(this_feature_x,window_shape_x)
#    
#        #print ('X_win',x_win.shape)
#        if b_Init==True:
#         multiFeatures_X_win=x_win;
#         b_Init=False;
#        else:    
#         multiFeatures_X_win=np.dstack((multiFeatures_X_win,x_win))
#    
#    
#    y_win = view_as_windows(y_, window_shape_y)  
#
#
#    if b_PrintShapes: print ('Gen Win Output X_win shape',multiFeatures_X_win.shape)
#    if b_PrintShapes: print ('Gen Win Output Y_win shape',y_win.shape)
#
#    #list of targets for differing prediction windows
#    l_y_ = []
#    for pred_win in l_d_pred_window:
#        
#        #print ('prediction window', pred_win)
#        this_y_ = y_win[:,sliding_window+pred_win-1]
#        
#       
#        l_y_.append(this_y_);
#        
#        print ('Gen Window final y sizes',this_y_.shape)
#    
#    return multiFeatures_X_win,l_y_
#
