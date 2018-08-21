#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:11:22 2018

@author: raiv
"""
#model class here

from __future__ import print_function
from math import sqrt
import numpy as np
import tensorflow as tf
import datamanager as dm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Defines Network model using tensorflow
class RNN_Model(object):
    def __init__(self, model_type,input_shape, params_dict):
         #self.network_shape = network_shape

         
         self.seq_length =params_dict['seq_length'] 
         self.train_count,self.seq_length,self.num_features=input_shape;
         
         
         self.learning_rate = params_dict['learning_rate']
#         self.decay_steps = decay_steps
#         self.decay_rate = decay_rate
#        
         self.lambda_loss_amount = params_dict['reg_parameter']
#         self.dropout_keep_prob = dropout_keep_prob
         self.inj_noise_std=params_dict['noise_std']
          
         
         self.training_epochs =params_dict['epoch']
         self.batch_size = params_dict['batch_size'] 
         self.num_batches = int(self.train_count / self.batch_size)
     
         self.model_type=model_type
         
         self.graph = tf.Graph()
         
         self.logs_path=dm.get_results_dir()  +'summaries' 
         
         dm.make_folder_in_results(self.logs_path)
         print ("log path",self.logs_path)
        
        
         if model_type == 'LSTM':
             print ('LSTM Architecture') 
             self.net_type='LSTM'   
             # RNN structure  

             self.create_RNN_FC_network(params_dict)
             
         elif    model_type == 'cLSTM':
             print ('cuda LSTM Architecture') 
             self.net_type='cLSTM'   
             # RNN structure  

             self.create_RNN_FC_network(params_dict) 
         
         elif model_type=='DNN':
              print ('DNN Arch')
              self.net_type='DNN'   
              self.create_DNN_network(params_dict)
              
         return    
	


    def create_DNN_network(self,params_dict):
        
         with tf.device('/device:GPU:0'):
              with self.graph.as_default():
                   # data tensors
                   self.X_tf = tf.placeholder(tf.float32, [None, self.seq_length,self.num_features]) # i/p batch size, Seq_Lgth, features
     
                   self.Y_tf = tf.placeholder(tf.float32, [None, 1]) # just one value out per input 
                  
                   
                   self.batch_size_tf = tf.placeholder(tf.int64) 
                   self.std_noise_tf=tf.placeholder(tf.float32)
                   self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_tf,self.Y_tf)).batch(self.batch_size_tf).repeat()
                   self.train_dataset=self.train_dataset.shuffle(buffer_size=10000)
                   #, (tf.TensorShape([None, self.seq_length,self.num_features]), tf.TensorShape([None,1])),(tf.float32, tf.float32)
                  
                   self.test_dataset = tf.data.Dataset.from_tensor_slices((self.X_tf,self.Y_tf)).batch(self.batch_size_tf)   
     			
                   self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                    self.train_dataset.output_shapes)
                   next_batch_op = self.iterator.get_next()
                   self.batch_X=next_batch_op[0]
                   self.batch_Y =next_batch_op[1]
                   self.batch_X_noised=self.gaussian_noise_layer(self.batch_X,self.std_noise_tf)
                               # make datasets that we can initialize separately, but using the same structure via the common iterator
                   self.training_data_init_op = self.iterator.make_initializer(self.train_dataset)
                               
                   self.test_data_init_op = self.iterator.make_initializer(self.test_dataset)
                   
                   
                   #architecture tensors 
                   self.inputLayer_flat = tf.contrib.layers.flatten(tf.reshape(self.batch_X_noised,[-1,self.seq_length,self.num_features]))
                   
        
                   self.hidden1 = tf.layers.dense(inputs=self.inputLayer_flat, units=params_dict['num_hidden1_units'], activation=tf.nn.relu)
                   self. hidden2 = tf.layers.dense(inputs=self.hidden1, units=params_dict['num_hidden2_units'], activation=tf.nn.relu)
                   self. hidden3 = tf.layers.dense(inputs=self.hidden2, units=params_dict['num_hidden3_units'], activation=tf.nn.relu)
                   
                   self.prediction = tf.layers.dense(inputs=self.hidden1, units=1)
     
    
                   # training parameters 
   
                   l2 = self.lambda_loss_amount * \
                                     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
                   
                  
                   self.loss=tf.reduce_mean(tf.losses.mean_squared_error(self.batch_Y, self.prediction)) + l2 # MSE loss
     
     
                   self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                
                   self.performance = tf.reduce_mean(tf.losses.mean_squared_error(self.batch_Y, self.prediction)) 
       
        
        
        
         return;
    def create_RNN_FC_network(self,params_dict):
         
         
         with tf.device('/device:GPU:0'):
              with self.graph.as_default():
     			# Input 
                   # @todo maybe a data object by itself ?
                   self.X_tf = tf.placeholder(tf.float32, [None, self.seq_length,self.num_features]) # i/p batch size, Seq_Lgth, features
     
                   self.Y_tf = tf.placeholder(tf.float32, [None, 1]) # just one value out per input 
                  
                   
                   self.batch_size_tf = tf.placeholder(tf.int64) 
                   self.std_noise_tf=tf.placeholder(tf.float32)
                   self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_tf,self.Y_tf)).batch(self.batch_size_tf).repeat()
                   self.train_dataset=self.train_dataset.shuffle(buffer_size=10000)
                  
                  
                   self.test_dataset = tf.data.Dataset.from_tensor_slices((self.X_tf,self.Y_tf)).batch(self.batch_size_tf)   
     			
                   self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                    self.train_dataset.output_shapes)
                   next_batch_op = self.iterator.get_next()
                   self.batch_X=next_batch_op[0]
                   self.batch_Y =next_batch_op[1]
                   self.batch_X_noised=self.gaussian_noise_layer(self.batch_X,self.std_noise_tf)
                               # make datasets that we can initialize separately, but using the same structure via the common iterator
                   self.training_data_init_op = self.iterator.make_initializer(self.train_dataset)
                               
                   self.test_data_init_op = self.iterator.make_initializer(self.test_dataset)
                                

                   # Dropout keep probability (set to 1.0 for validation and test)
                   self.keep_prob = tf.placeholder(tf.float32)
     			# FC Variables
                   
                   self.num_hidden = params_dict ['num_hid_units'] # nb of neurons inside the neural network
                   self.num_rnn_layers=params_dict['num_rnn_layers'];
                   
                   self.W = {
                             'hidden': tf.Variable(tf.truncated_normal([self.num_features, self.num_hidden])),
                             'output': tf.Variable(tf.truncated_normal([self.num_hidden, 1])),
                      }
                   
                   self.biases = {
                             'hidden': tf.Variable(tf.random_normal([self.num_hidden])),
                     'output': tf.Variable(tf.random_normal([1])),
                      }
                 
                   # Predictions for the training, validation, and test data
                   scope = self.net_type 
                   
                   #print ("Cell input keep: %f  output keep: %f state keep %f" %(flt_in_keep,flt_out_keep,flt_state_keep))
                 
                   
                   if self.net_type == 'LSTM':   
                        print ('Simple LSTM')
                        self.rnn_layers = tf.contrib.rnn.MultiRNNCell([self.get_LSTM(self.num_hidden) for _ in range(self.num_rnn_layers)])
                    
                   elif self.net_type == 'cLSTM':
                        print ('cuda LSTM Architecture')
                        self.rnn_layers = tf.contrib.rnn.MultiRNNCell([self.get_cudnnLSTM(self.num_hidden) for _ in range(self.num_rnn_layers)])
                 
                   elif self.net_type == 'GRU':
                        print ('GRU Architecture')
                        self.rnn_layers = tf.contrib.rnn.MultiRNNCell([self.get_GRU(self.num_hidden) for _ in range(self.num_rnn_layers)])
                        
                        
                     #  loss and L2
                   l2 = self.lambda_loss_amount * \
                                     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
                   
                    
                   self.prediction = self.get_predictions(self.batch_X_noised,scope)                  
                   self.loss=tf.reduce_mean(tf.losses.mean_squared_error(self.batch_Y, self.prediction)) + l2 # MSE loss
                   tf.summary.scalar('loss',self.loss)
     
                   self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                
                   self.performance = tf.reduce_mean(tf.losses.mean_squared_error(self.batch_Y, self.prediction)) 
     
     			# Global Step for learning rate decay
                   #global_step = tf.Variable(0)
                   #self.initial_learning_rate = initial_learning_rate #tf.train.exponential_decay(self.initial_learning_rate, global_step, self.decay_steps, self.decay_rate)
     
     			# Passing global_step to minimize() will increment it at each step.

                   merged = tf.summary.merge_all()
                   
#                   train_writer = tf.summary.FileWriter(self.logs_path + '/train',
#                                      sess.graph)
#                   test_writer = tf.summary.FileWriter(self.logs_path + '/test')
   
    
    def get_predictions(self,batch_X_tf, scope):
         '''model a LSTM Network,
              it stacks 2 LSTM layers, each layer has n_hidden=32 cells
              and 1 output layer, it is a full connet layer
              argument:
                   feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
                   config: class containing config of network
                   return:
              : matrix  output shape [batch_size,1]
         '''               
    
         with tf.variable_scope (scope):
         # stack lstms#    # Stack two LSTM layers, both layers has the same shape   
         
         
              batch_X_tf_list = self.reformat_for_static_rnn(batch_X_tf)

          

              # static rnn inputs: A length T list of inputs, 
              #each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
       
              rnn_outputs_tf_list, _ = tf.nn.static_rnn(self.rnn_layers, batch_X_tf_list, dtype=tf.float32)
       
             # print ("outputs len",len(rnn_outputs_tf_list))  # timesteps       
       
              last_out_rnn_tf= rnn_outputs_tf_list[self.seq_length-1];        
               #print ('Last outputs shape of LSTM cell ',last_out_rnn_tf.shape)        # batch_size, state size
        
       
        
               #rehape to have n_statesize col. # new shape [batch_sz,state_size]
              last_out_rnn_tf = tf.reshape(last_out_rnn_tf, [-1, self.num_hidden])         
              #print ("reshaped output ",last_out_rnn_tf.shape )#
        
               #FC gives one col . ie 1 value for each example in batch for each timestep  
               # tsteps* batch_size outputs, but tsteps= 1 for sliding window version

              predictions = tf.matmul(last_out_rnn_tf, self.W['output']) + self.biases['output']
               #print ('post FC  multiply; shape',last_out_rnn_tf.get_shape()) #  * batch_size , 1
        
        
    
      
         return predictions


    
    def reformat_for_static_rnn(self,batch_X_tf):
        
        
        # dynamic rnn input: this must be a Tensor of shape: [batch_size, max_time, ...]
        #outputs_, _ = tf.nn.dynamic_rnn(lsmt_layers, batch_X, dtype=tf.float32)
        #outputs_=tf.transpose(outputs_,[1,0,2])
        batch_X_tf=tf.transpose(batch_X_tf,[1,0,2]);
        batch_X_tf=tf.reshape(batch_X_tf,[-1,self.num_features]) # flatten 
        batch_X_tf_list=tf.split(batch_X_tf,self.seq_length ,axis=0)
            
        #print(' Length of feature mat after Split ',len(batch_X_list))
        #print ("Each input tensor shape", batch_X_list[0].get_shape())                                                           
         
    
        return batch_X_tf_list
    
    
    



    def train(self,_trainX,_trainY,_testX,_testY):
    
    
          # here starts one trial
          
       	 with tf.Session(graph=self.graph) as sess:
              init=tf.global_variables_initializer()
              sess.run(init)
              
              #print("Initialized ")
              #self.writer = tf.train.SummaryWriter(self.logs_path, graph=tf.get_default_graph())

              for iter_epochs in range (self.training_epochs):
                    
                   for start, end in zip(range(0, self.train_count, self.batch_size),
                                                          range(self.batch_size, self.train_count + 1, self.batch_size)):
                                        
                        # shuffle batch before training   
                        batch_X_shuffled,batch_Y_shuffled=dm.shuffle_in_unison(_trainX[start:end],_trainY[start:end])
                        #print ('batch input shapes',_trainX[start:end].shape,_trainY[start:end].shape)
     
                        batch_X_shuffled_noised = batch_X_shuffled+(self.inj_noise_std*np.random.randn(*batch_X_shuffled.shape));
                        sess.run(self.optimizer, feed_dict={self.X_tf:batch_X_shuffled_noised,self.Y_tf: batch_Y_shuffled})
                                       
                        #print loss every 20 epochs
                   if iter_epochs % 5 == 0:
                        [loss_out_this_epoch] = sess.run([self.performance], feed_dict={self.X_tf: batch_X_shuffled, self.Y_tf: batch_Y_shuffled})
                        print ("Epoch ",str(iter_epochs) +" Last Batch MSE ",(loss_out_this_epoch))
                             
              #print('Saving noised and shuffled input ', )
              dm.save_as_mat(['inputs','shuffled_input'],dict([ ('batch_X_shuffled_noised', batch_X_shuffled_noised), ('batch_X_shuffled', batch_X_shuffled)]))
              
              
              predictions, ms_error = sess.run([self.prediction, self.performance], feed_dict={self.X_tf: _testX, self.Y_tf: _testY})                                             
 
     
     
              #reset graph after each trial ? or after test predictions
              #tf.reset_default_graph() 
              #sess.close()
              return predictions, ms_error
              
    def train_data(self,_trainX,_trainY,_testX,_testY):
    
    
    
    

         bSaveInput = True;
          # here starts one trial
         #config = tf.ConfigProto()
         config=tf.ConfigProto(log_device_placement=True)
         #config.gpu_options.per_process_gpu_memory_fraction = 0.6
         config.gpu_options.allow_growth = True 
       	 with tf.Session(config=config,graph=self.graph) as sess:
              init=tf.global_variables_initializer()
              print ("data iterator initialized to training" )
              sess.run(self.training_data_init_op, feed_dict = {self.X_tf : _trainX, self.Y_tf: _trainY,self. batch_size_tf:self.batch_size} )
              sess.run(init)
              
              #print("Initialized ")
              #self.writer = tf.train.SummaryWriter(self.logs_path, graph=tf.get_default_graph())

              for iter_epochs in range (self.training_epochs):
                   
                   for _ in range(self.num_batches):
                        
                        if bSaveInput:
                             #print('Saving noised and shuffled input ', )
                             [batch_X_shuffled,batch_X_shuffled_noised]=sess.run([self.batch_X,self.batch_X_noised],feed_dict={self.std_noise_tf:self.inj_noise_std})
                             dm.save_as_mat(['inputs','shuffled_input'],dict([ ('batch_X_shuffled_noised', batch_X_shuffled_noised), ('batch_X_shuffled', batch_X_shuffled)]))
                             bSaveInput = False;
                             
                        sess.run([self.optimizer, self.loss],feed_dict={self.std_noise_tf:self.inj_noise_std})

                        #print loss every 20 epochs
                   if iter_epochs % 5 == 0:
                       # [loss_out_this_epoch] = sess.run([self.performance], feed_dict={self.std_noise_tf:self.inj_noise_std})
                        print ("Epoch ",str(iter_epochs) +" Last Batch MSE ",(sess.run([self.performance], feed_dict={self.std_noise_tf:self.inj_noise_std})))
                             
              
              
              
              print ("data iterator initialized to test set")
              sess.run(self.test_data_init_op, feed_dict = {self.X_tf : _testX, self.Y_tf: _testY,self. batch_size_tf:_testY.shape[0]} )
              predictions, ms_error = sess.run([self.prediction, self.performance], feed_dict={self.std_noise_tf:self.inj_noise_std})                                             
 
     
     
              #reset graph after each trial ? or after test predictions
              #tf.reset_default_graph() 
              sess.close()
              return predictions, ms_error
                      
                                  
		

        
    @staticmethod    
    def get_LSTM(n_hidden):
         
         b_peep=False;
         
         lstm_cells=tf.contrib.rnn.LSTMCell(num_units=n_hidden,use_peepholes=b_peep, state_is_tuple=True,forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
         
         #lstm_cells=tf.contrib.cudnn_rnn.CudnnLSTM(num_units=n_hidden,use_peepholes=b_peep, state_is_tuple=True,forget_bias=1.0, reuse=tf.get_variable_scope().reuse)

         lstm_cells = tf.nn.rnn_cell.DropoutWrapper(lstm_cells, input_keep_prob=1.0, output_keep_prob=1.0,state_keep_prob=1.0)
         return lstm_cells

    @staticmethod
    def get_cudnnLSTM(n_hidden):
         print (" cUDA compatible LSTM cell")
         #lstm_cells=tf.contrib.cudnn_rnn.CudnnLSTM(num_units=n_hidden,num_layers=1)
 
         lstm_cells=tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=n_hidden)
         #lstm_cells = tf.nn.rnn_cell.DropoutWrapper(lstm_cells, input_keep_prob=flt_in_keep, output_keep_prob=flt_out_keep,state_keep_prob=flt_state_keep)
         return lstm_cells
    @staticmethod
    def gaussian_noise_layer(input_layer, std):
         print ("Gaussian Noise added",tf.shape(input_layer))
    
         noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
          #batch_X_shuffled_noised = batch_X_shuffled+(sig_noise_input*np.random.randn(*batch_X_shuffled.shape));

    
         return input_layer + noise
 
     
            
    def printConfig(self):
        
        print ('Config deets\n')
                
        print ('Number of examples',self.train_count,'each with No. of timesteps',self.seq_length )
        print ('Number of features', self.num_features)
        print ('Batch Size',self.batch_size)
        print ('Typr of Network ',self.net_type)
        print ('Number of RNN layers',self.num_rnn_layers)
        print ('Hidden Units',self.num_hidden)
        print ('  ')
              
        return ;

''' ###########################################################################
'''   
     
     #    def bi_RNN_Network(batch_X, config,scope):
#    """model bidirectional LSTM/GRU Network,
#      it stacks 2 LSTM layers, each layer has n_hidden cells for fowd and bck cells
#       and 1 output layer, it is a full connet layer
#      argument:
#        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
#        config: class containing config of network
#        
#      Network:
#          
#       tf.nn.static_bidirectional_rnn( cell_fw,  cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None,  scope=None) 
#
#      return:
#              : matrix  output shape [batch_size,1]
#
#      #   """ 
#
#    print ("BI Directional RNN")
#    with tf.variable_scope (scope):
#         # stack lstms#    # Stack two LSTM layers, both layers has the same shape    
#         
#        if arch_nn == 'LSTM':
#            print ('LSTM Architecture')
#            rnn_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell(config.n_hidden) for _ in range(config.n_rnn_layers)])
#            
#        elif arch_nn == 'TF-LSTM':
#            print ('Time Freq LSTM Architecture')
#            rnn_layers = tf.contrib.rnn.MultiRNNCell([TFlstm_cell(config.n_hidden) for _ in range(config.n_rnn_layers)])
#            
#        elif arch_nn== 'GRU':
#            print ('GRU Architecture')
#            rnn_layers = tf.contrib.rnn.MultiRNNCell([GRU_cell(config.n_hidden) for _ in range(config.n_rnn_layers)])
#      
#        
#                  
#        print ("Cell input keep: %f  output keep: %f state keep %f" %(flt_in_keep,flt_out_keep,flt_state_keep))
#
#        batch_X=tf.transpose(batch_X,[1,0,2]);
#        batch_X=tf.reshape(batch_X,[-1,config.n_features]) # flatten 
#        l_batch_X_tf=tf.split(batch_X,config.n_timesteps,axis=0)
#            
#        #print(' Length of feature mat after Split ',len(batch_X_list))
#        #print ("Each input tensor shape", batch_X_list[0].get_shape())                                                           
#         
#       # static bi rnn inputs: A length T list of inputs, #each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
#       #            outputs: A tuple (outputs, output_state_fw, output_state_bw) where: outputs is a length T list of outputs (one for each input), 
#        l_outputs_tf,_,_ = tf.nn.static_bidirectional_rnn(rnn_layers,rnn_layers, l_batch_X_tf, dtype=tf.float32)
#       
#        print ("outputs len",len(l_outputs_tf))  # timesteps       
#       
#        last_out_rnn_tf= l_outputs_tf[config.n_timesteps-1];        
#        print ('Last outputs shape of LSTM cell ',last_out_rnn_tf.shape)        # batch_size, state size * 2
#        
#       
#        
#        #rehape to have n_statesize col. # new shape [batch_sz,state_size_fw+state_size_bw] ? is this needed?
#        last_out_rnn_tf = tf.reshape(last_out_rnn_tf, [-1, config.n_hidden*2])         
#        print ("reshaped output ",last_out_rnn_tf.shape )#
#        
#        #FC gives one col . ie 1 value for each example in batch for each timestep  
#        # tsteps* batch_size outputs, but tsteps= 1 for sliding window version
#
#        last_out_rnn_tf = tf.matmul(last_out_rnn_tf, config.W['output_bi']) + config.biases['output_bi']
#        print ('post FC  multiply; shape',last_out_rnn_tf.get_shape()) #  * batch_size , 1
#        
#        
    
      
#    return [last_out_rnn_tf]