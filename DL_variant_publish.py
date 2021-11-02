# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2021
Code for charging station occupancy prediction using DL variants.

More detail: see the paper: 
Ma, TY, Faye, S. (2021) Multistep Electric Vehicle Charging Station Occupancy
 Prediction using Hybrid LSTM Neural Networks. arXiv:2106.04986
    
@author: Tai-yu MA
"""

import math
import time
import pandas as pd
import numpy  as np 
from matplotlib import pyplot
from tensorflow import keras
from numpy import array 
from keras.models import Sequential 
from keras.layers import LSTM,GRU,ConvLSTM2D
from keras.layers import RepeatVector
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from keras.layers import BatchNormalization 
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
from keras.models import Model 
from keras.layers import Input
from keras.layers.merge import concatenate

 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

## read data
def read_data(string,model_id, t_win, n_steps_in,n_steps_out,n_features):
    
    Z = pd.read_csv(string)
    Z=Z.to_numpy()
     
    Z.shape 
    if model_id=='LSTM' or model_id=='GRU' or model_id=='BiLSTM' or model_id=='StackedLSTM' or model_id=='Conv1D' or  model_id=='Decoder':
        X, y = split_sequences(Z, n_steps_in, n_steps_out ) 
    elif model_id=='CNN_LSTM' or model_id=='CNNDecoder': 
       X, y = split_sequences(Z, t_win, n_steps_out )
       X = X.reshape((X.shape[0], n_steps_in, n_steps_out, n_features))
    elif model_id=='ConvLSTM':
       X, y = split_sequences(Z, t_win, n_steps_out )
       X = X.reshape((X.shape[0], n_steps_in, 1, n_steps_out, n_features))

    n_train=int(0.7*len(X)) 
    
    X_train=X[0: n_train,];         y_train = y[0:n_train,]
    X_test =X[n_train: len(X),];    y_test  = y[n_train:len(X),]
    X_train.shape
    X_test.shape 
    X_train[0,]
    
    return X_train,y_train,X_test,y_test 

#########
# LSTM
##########
def fit_model_LSTM(res ,_iter, X_train,y_train,X_test,y_test,t_win,
                   n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
  
    model = Sequential() 
    model.add(LSTM(n_n_lstm,   input_shape=(n_steps_in, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout)) 
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras.utils.plot_model(model, show_shapes=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
     
    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,n_epoch,
                  bat_size, _mean1 ]  
    return res 


#########
# GRU
##########
def fit_model_GRU(res ,_iter, X_train,y_train,X_test,y_test,t_win,
                   n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
  
    model = Sequential() 
    model.add(GRU(n_n_lstm,   input_shape=(n_steps_in, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout)) 
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras.utils.plot_model(model, show_shapes=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
     
    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,n_epoch,
                  bat_size, _mean1 ]  
    return res 
    
    
 
#########
# bi-LSTM
##########
 
def fit_model_BiLSTM(res ,_iter, X_train,y_train,X_test,y_test,t_win,
                   n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
  
    model = Sequential() 
    model.add(Bidirectional(LSTM(n_n_lstm),  input_shape=(n_steps_in, n_features))) 
    model.add(Dense(100, activation='relu'))    
    model.add(Dropout(dropout))   
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras.utils.plot_model(model, show_shapes=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
     
    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,n_epoch,
                  bat_size, _mean1 ]  
    return res 
    

#########
# Stacked_LSTM
##########
def fit_model_StackedLSTM(res ,_iter, X_train,y_train,X_test,y_test,t_win, 
              n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
    
  
    model = Sequential() 
    model.add(LSTM(nf_1, return_sequences=True, input_shape=(n_steps_in, n_features)))    
    model.add(LSTM(nf_2))    
    model.add(Dense(100, activation='relu'))    
    model.add(Dropout(dropout))
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #keras.utils.plot_model(model, show_shapes=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
     
    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,n_epoch,
                  bat_size, _mean1 ]  
    return res 
    

#########
# Conv1D
##########
def fit_model_Conv1D(res ,_iter, X_train,y_train,X_test,y_test,t_win, 
              n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
     
    model = Sequential() 
    model.add(Conv1D(filters=nf_1, kernel_size=ker_size, 
             activation='relu', input_shape=(n_steps_in, n_features))) 
    model.add(Conv1D(filters=nf_2, kernel_size=ker_size, activation='relu')) 
    model.add(MaxPooling1D(pool_size=po_size))
    model.add(Flatten())   
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout)) 
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # model.summary()
    keras.utils.plot_model(model, show_shapes=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)

    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,n_epoch,
                  bat_size, _mean1 ]  
    return res 
    

################
# model CNN_LSTM
###############
def fit_model_CNN_LSTM(res ,_iter, X_train,y_train,X_test,y_test,t_win, 
              n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
   
    model = Sequential() 
    model.add(TimeDistributed(Conv1D(filters=nf_1, kernel_size=ker_size, 
            padding='same',activation='relu'), input_shape=(n_steps_in, n_steps_out, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
    model.add(TimeDistributed(Conv1D(filters=nf_2, kernel_size=ker_size, padding='same',activation='relu'))) 
    model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
    model.add(TimeDistributed(Flatten())) 
    model.add(LSTM(n_n_lstm)) 
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))   
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # model.summary()
    keras.utils.plot_model(model, show_shapes=True)
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
       
    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,
                  n_epoch,bat_size, _mean1 ]  
    return res 
    

################
# model ConvLSTM
############### 
def fit_model_ConvLSTM(res ,_iter, X_train,y_train,X_test,y_test,t_win, 
              n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
              po_size,n_n_lstm,dropout,n_epoch,bat_size):
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=nf_1, kernel_size=(1,ker_size), 
            activation='relu', input_shape=(n_steps_in, 1, n_steps_out, n_features)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))  
    model.add(Dense(n_steps_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=27,verbose=2)

    temp = model.predict(X_test, verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)     
    res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,
                  n_epoch,bat_size, _mean1 ]  
    return res 


def run(model_id, n_steps_in,n_steps_out,n_features,n_epoch,n_trivals,n_out,
            nf_1,nf_2,ker_size,po_size,n_n_lstm,dropout,bat_size): 
    
    t_win=n_steps_in*n_steps_out        
    n_station=9
    string='U:/DL/data_chg_all_feature_'   
    station=[string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv'
             ,string+'6.csv',string+'7.csv',string+'8.csv',string+'9.csv']
  
    res_all=[]
    for s in range(n_station):         
        X_train,y_train,X_test,y_test =read_data(station[s],model_id, t_win, 
                                n_steps_in,n_steps_out,n_features)
        res=np.zeros([n_trivals,n_out])
        X_train.shape
        y_train.shape 
        for _iter in range(n_trivals):   
            if model_id=='LSTM':
                res=fit_model_LSTM(res,_iter, X_train,y_train,X_test,y_test ,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size) 
            elif model_id=='BiLSTM':                    
               res=fit_model_BiLSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size)     
            elif model_id=='StackedLSTM':                    
               res=fit_model_StackedLSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size)  
            elif model_id=='Conv1D':                    
               res=fit_model_Conv1D(res,_iter, X_train,y_train,X_test,y_test,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size) 
            elif model_id=='CNN_LSTM':
                 res=fit_model_CNN_LSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size)   
            elif model_id=='GRU':
                 res=fit_model_GRU(res,_iter, X_train,y_train,X_test,y_test,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size) 
            elif model_id=='ConvLSTM':
                 res=fit_model_ConvLSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
                           n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
                           ker_size, po_size,n_n_lstm,dropout,n_epoch,
                           bat_size) 
               
        _mean = np.mean(res[:,-1:],axis=0)
        _std  = np.std(res[:,-1:],axis=0)
        res_all.append([_mean,_std])
        
    temp=[]
    for i in range(n_station):            
        temp.append(res_all[i][0])
        
    accuracy_avg1=np.mean(temp, axis=0)
    accuracy_avg2=np.mean(temp, axis=1)  
    return accuracy_avg1, accuracy_avg2, res_all

   
def main():
   
    n_steps_in =3  # t-3,t-2,t-1
    n_features = 148 # 148 for all features
    n_steps_out =6# num of predicted steps, if n_steps_out =1 or 3 po_size needs to be 1
    n_epoch_global=15
    n_trivals=10
    n_out=9 
    nf_1=32
    nf_2=16
    ker_size=4
    po_size=2
    n_n_lstm=16
    dropout=0.4
    bat_size= 30
    accuracy_avg_1=[]
    accuracy_avg_2=[]
    flag_sensitivity=0
    model_id='LSTM'
    #model_id='GRU'
    # model_id='BiLSTM'
    #model_id='StackedLSTM'
    #model_id='Conv1D'   
    #model_id='CNN_LSTM' 
   # model_id='ConvLSTM'
    if  model_id=='ConvLSTM' or  model_id=='Conv1D'  :
        ker_size=1       

    if flag_sensitivity==1: #sensitivity analysis
        parameter = [11,12,13,14,15,16,17,18,19,20] #,13,14,15,16,17,18,19,20
        for i in range(len(parameter)):
            avg1,avg2,res_all =  run(model_id,n_steps_in,n_steps_out,n_features,
                           parameter[i],n_trivals,n_out,nf_1,nf_2,ker_size,
                           po_size,n_n_lstm,dropout,bat_size)  
            accuracy_avg_1.append(avg1)
            accuracy_avg_2.append(avg2)            
    else:   avg1,avg2,res_all =  run(model_id,n_steps_in,n_steps_out,n_features,
                            n_epoch_global,n_trivals,n_out,nf_1,nf_2,ker_size,
                            po_size,n_n_lstm,dropout,bat_size)   
    accuracy_avg_1.append(avg1)
    accuracy_avg_2.append(avg2)
    
    ## output results    
    print('model: ', model_id)
    print('sensitivity_flag = ', flag_sensitivity) 
    if flag_sensitivity==1:
        print('parameter : ', parameter) 
    print('n_step out: ', n_steps_out)
    print('n_epoch,n_trivals, nf_1,nf_2,po_size,n_n_lstm,dropout,bat_size', 
          n_epoch_global,n_trivals, nf_1,nf_2,po_size,n_n_lstm,dropout,bat_size)       
    print('accuracy_avg_1: ',accuracy_avg_1)
    print('accuracy_avg_2: ',accuracy_avg_2)
    return res_all
    
res_all = main()
 
    