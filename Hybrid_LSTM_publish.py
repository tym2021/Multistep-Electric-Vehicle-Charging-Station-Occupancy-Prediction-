# -*- coding: utf-8 -*-
"""
Created on Nov 2, 2021
Code for charging station occupancy prediction using hybrid LSTM 
and Machine Learning approaches.
More detail: see the paper: 
Ma, TY, Faye, S. (2021) Multistep Electric Vehicle Charging Station Occupancy
 Prediction using Hybrid LSTM Neural Networks. arXiv:2106.04986
    
@author: Tai-yu MA
"""

import math
import time
import pandas as pd
import numpy  as np 
from numpy import array
from matplotlib import pyplot
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import LSTM 
from keras.layers import Dense,Dropout  
from keras.models import Model 
from keras.layers import Input
from keras.layers.merge import concatenate

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
 
 
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


## read data for machine learning approaches
def read_data_ML(string):
    Z = pd.read_csv(string)
    Z=Z.to_numpy()
     
    Z.shape  
    n_train=int(0.7*len(Z))
    
    X_train=Z[0: n_train,0:-1];         y_train = Z[0:n_train,-1]
    X_test =Z[n_train: len(Z),0:-1];    y_test  = Z[n_train:len(Z),-1]
    X_train.shape
    X_test.shape 
    X_train[0,] 
    
    return X_train,y_train,X_test,y_test 


## read data for hybrid LSTM
def read_data(string,string2,model_id, n_steps_in,n_steps_out,n_features):
     
    Z = pd.read_csv(string)
    Z=Z.to_numpy()
     
    Z.shape 
    X, y = split_sequences(Z[:,3:5], n_steps_in, n_steps_out )
        #X, y = split_sequences(Z, n_steps_in, n_steps_out )
    
    n_train=int(0.7*len(X))
    Z1 = pd.read_csv(string2)#col1 weekday,col2 weekend
    Z1 = Z1.to_numpy()
    Z1 = Z1.transpose()
    Z2 =np.concatenate((Z1,Z1),axis=1) 
    X2 = np.zeros([len(Z),3+144],float)  
  
    for i in range(len(Z)-n_steps_in): 
     if Z[i+n_steps_in-1,-1]==0:      
       #  qq=np.array(Z2[0][Z[i+n_steps_in-1,0]:Z[i+n_steps_in-1,0]+n_steps_out])
          qq=np.array(Z2[0][0:144])
          X2[i]=np.append(Z[i+n_steps_in-1][0:3],qq) 
     else:            
        # qq=np.array(Z2[1][Z[i+n_steps_in-1,0]:Z[i+n_steps_in-1,0]+n_steps_out])
         qq=np.array(Z2[1][0:144])
         X2[i]=np.append(Z[i+n_steps_in-1][0:3],qq) 
    
    X_train=X[0: n_train,];         y_train = y[0:n_train,]
    X_test =X[n_train: len(X),];    y_test  = y[n_train:len(X),]
    X_train.shape
    X_test.shape 
    X_train[0,]
    X2_train=X2[0: n_train,]; X2_test =X2[n_train: len(X),];  
    X2_train.shape
    X2_test.shape
    X2_train[0,]  
    
    return X_train,y_train,X_test,y_test,X2_train,X2_test
 
    
############
# Mix_LSTM
##############
def fit_model_MixLSTM(res_F1,res ,_iter, X_train,y_train,X_test,y_test,X2_train,X2_test,
                   n_steps_in,n_steps_out,n_features, n_n_lstm,dropout,n_epoch,bat_size):
    
     
    input1 = keras.Input(shape=(n_steps_in, n_features))
    input2 = keras.Input(shape=(147,))  
    model_LSTM=LSTM(n_n_lstm)(input1)
    model_LSTM=Dropout(dropout)(model_LSTM)
    model_LSTM=Dense(18, activation='relu')(model_LSTM)
   
    meta_layer = keras.layers.Dense(147, activation="relu")(input2)
    meta_layer = keras.layers.Dense(64, activation="relu")(meta_layer)    
    meta_layer = keras.layers.Dense(32, activation="relu")(meta_layer)
    model_merge = keras.layers.concatenate([model_LSTM, meta_layer])
    model_merge = Dense(100, activation='relu')(model_merge)
    model_merge = Dropout(dropout)(model_merge)    
    output = Dense(n_steps_out, activation='sigmoid')(model_merge)
    model = Model(inputs=[input1, input2], outputs=output) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras.utils.plot_model(model, show_shapes=True)
    #print(model.summary()) 
    model.fit([X_train, X2_train], y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
    
    temp = model.predict([X_test,X2_test], verbose=2)
    m,n=temp.shape 
    t_target = n_steps_out
       
    yhat=np.zeros((m,t_target))
    y_obs=np.array(y_test[0:m,0:t_target])
    scores1= np.zeros(m)
    scores_F1= np.zeros([m,3],float)
    for i in np.arange(m):  
        for j in np.arange(t_target):  
            if temp[i][j]>=0.5:
                yhat[i][j]=1           
        val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
        scores_F1[i,0] = precision_score(y_obs[i,:], yhat[i,],zero_division=1)
        scores_F1[i,1] = recall_score(y_obs[i,:], yhat[i,],zero_division=1)
        scores_F1[i,2] = f1_score(y_obs[i,:], yhat[i,],zero_division=1)
        scores1[i]=val       
     
    _mean1 = np.mean(scores1)      
    _mean_F1 = np.mean(scores_F1,axis=0)  
    res[_iter,:]=[ n_n_lstm,dropout,n_epoch, bat_size, _mean1 ]  
    res_F1[_iter,:]=_mean_F1
    return res_F1, res     

    
def run_ML(model_id,n_steps_out):
     
    n_station=9 
    string='U:/DL/data_chg_ML_'
    station=[string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv',string+'6.csv',string+'7.csv',string+'8.csv',string+'9.csv']

    if    model_id=='logistic': idx_model = LogisticRegression()
    elif  model_id=='svc':      idx_model = SVC()
    elif  model_id=='RF':       idx_model = RandomForestClassifier() 
    elif  model_id=='Ada':      idx_model = AdaBoostClassifier() 
    
    #models=[m_logit,m_KNN,m_SVC,m_RF,m_Bag,m_ada]
    models=[idx_model]
    rng_s_1=[s for s in range(3)]    
    rng_s_2=[s for s in range(6)]
    rng_s_3=[s for s in range(12)]
    rng_s_4=[s for s in range(24)]
    rng_s_5=[s for s in range(36)]
     
    step_set=[rng_s_1,rng_s_2,rng_s_3,rng_s_4,rng_s_5] 
    
    vec_mean_metrics=[]
    #res_all=[] # activate for one-step
    for s in range(n_station):         
        X_train,y_train,X_test,y_test =read_data_ML(station[s])
        res_all=[] #activate for multi-steps
        for mm in models:
            mm.fit(X_train, y_train)   
            # case n_step_out=1
            # predicted = mm.predict(X_test)    
            # _acc   =accuracy_score(y_test,predicted)
            # _pre   = precision_score(y_test,predicted)
            # _recall=recall_score(y_test,predicted)
            # _f1    = f1_score(y_test,predicted)
            # res_all.append([_acc, _pre,_recall,_f1])
            # mean_metrics = np.mean(res_all,axis=0)  
            #end case one step
           
            t_target = n_steps_out
            
            m,n=X_test.shape
            yhat=np.zeros([m,t_target])            
            y_obs=np.zeros([m,t_target])
            for kk in range(m-t_target) :
                y_obs[kk,:]=y_test[kk:kk+t_target]
                 
            scores1= np.zeros(m,float)
            scores_F1= np.zeros([m,3],float)            
            n_sample=m-n_steps_out     
            for i in range(n_sample): 
                X_test_temp=X_test.copy();  
                X_test_temp=np.append(X_test_temp,[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],],0)#dummy
                for j in range(n_steps_out):   
                    temp11=X_test_temp[i+j,:].reshape(1, -1)                    
                    yhat[i,j] = mm.predict(temp11) 
                    #if i+j+3<m:
                    rng1=[ i+j+3 -_ii for _ii in range(0, 3) ]
                    rng2=[ _jj for _jj in range(-3,0) ] # only t-3,-2,-1 are considered
                    X_test_temp[rng1,rng2]=yhat[i,j]  
                  
                res_temp=[]
                for rr in step_set: 
                    _acc   = accuracy_score(y_obs[i,rr], yhat[i,rr])
                    _pre   =precision_score(y_obs[i,rr], yhat[i,rr],zero_division=1)
                    _recall= recall_score(y_obs[i,rr], yhat[i,rr],zero_division=1)
                    _f1    = f1_score(y_obs[i,rr], yhat[i,rr],zero_division=1)
                    res_temp=np.append(res_temp,[_acc, _pre,_recall,_f1],0)
                
                res_all.append(res_temp) 
                
            _mean_metrics = np.mean(res_all,axis=0)   
            vec_mean_metrics.append(_mean_metrics)
            
    return  vec_mean_metrics
    
def run(model_id, n_steps_in,n_steps_out,n_features,n_epoch,n_trivals,n_out,
            n_n_lstm,dropout,bat_size): 
     
    
    t_win=n_steps_in*n_steps_out        
    n_station=9
    string='U:/DL/data_chg_'
    string2='U:/DL/data_chg_pred_occ_t_'
    station=[string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv',string+'6.csv',string+'7.csv',string+'8.csv',string+'9.csv']
    station2=[string2+'1.csv',string2+'2.csv',string2+'3.csv',string2+'4.csv',string2+'5.csv',string2+'6.csv',string2+'7.csv',string2+'8.csv',string2+'9.csv']
  
    res_all=[];res_all_F1=[]
    for s in range(n_station):         
        X_train,y_train,X_test,y_test,X2_train,X2_test=read_data(station[s],station2[s],model_id,  
                                n_steps_in,n_steps_out,n_features)
        res=np.zeros([n_trivals,n_out])
        res_F1=np.zeros([n_trivals,3])
        X_train.shape
        y_train.shape 
        for _iter in range(n_trivals):   
            if model_id=='Mix_LSTM':
                 res_F1,res=fit_model_MixLSTM(res_F1,res,_iter, X_train,y_train,X_test,y_test,X2_train,X2_test, 
                           n_steps_in,n_steps_out,n_features, n_n_lstm,dropout,n_epoch,
                           bat_size)  
               
        _mean = np.mean(res[:,-1:],axis=0)
        _std  = np.std(res[:,-1:],axis=0)
        res_all.append([_mean,_std])
       
        _mean_F1 = np.mean(res_F1,axis=0)
        _std_F1  = np.std(res_F1 ,axis=0)
        res_all_F1.append([_mean_F1])
        
    temp=[];temp1=[]
    for i in range(n_station):            
        temp.append(res_all[i][0])     
        
    accuracy_avg1=np.mean(temp, axis=0)
    accuracy_avg2=np.mean(temp, axis=1) 
    avg_metrics_prec_recall_F1=np.mean(res_all_F1, axis=0)
    
    return accuracy_avg1, accuracy_avg2, res_all,res_all_F1,avg_metrics_prec_recall_F1


def main():
   
    n_steps_in =12  # input y sequence for LSTM cell
    n_features = 1  # one feature for the input of the LSTM cell
    n_steps_out =6# num of predicted steps. 
    n_epoch_global=15
    n_trivals=10
    n_out=5  
    n_n_lstm=36
    dropout=0.2
    bat_size= 30
    accuracy_avg_1=[]
    accuracy_avg_2=[]
    flag_sensitivity=0  
    model_id='Mix_LSTM' 
    flag_ML=0 # to run machine learning models, set  flag_ML=1 otherwise 0
    # for ML, we set n_steps_out=36 as we compute the predcition for all forecasting cases 
    if  flag_ML==1:
        n_steps_out=36
    #choose the ML model to test
    
    #model_id='logistic'
    #model_id='svc'
    #model_id='RF'
    #model_id='Ada'
    
    if flag_ML==0:
        if flag_sensitivity==1:
            parameter = [11,12] #,13,14,15,16,17,18,19,20
            for i in range(len(parameter)):
                avg1,avg2,res_all, res_all_F1=  run(model_id,n_steps_in,n_steps_out,n_features,
                               parameter[i],n_trivals,n_out,n_n_lstm,dropout,bat_size)  
                accuracy_avg_1.append(avg1)
                accuracy_avg_2.append(avg2)            
        else:   
            avg1,avg2,res_all, res_all_F1, avg_metrics_prec_recall_F1=  run(model_id,n_steps_in,n_steps_out,n_features,
                                n_epoch_global,n_trivals,n_out,n_n_lstm,dropout,bat_size)  
            accuracy_avg_1.append(avg1)
            accuracy_avg_2.append(avg2) 
            
        print('model: ', model_id)
        print('sensitivity_flag = ', flag_sensitivity) 
        if flag_sensitivity==1:
            print('parameter : ', parameter) 
        print('n_step out: ', n_steps_out)
        print('n_epoch,n_trivals, n_n_lstm,dropout,bat_size', 
              n_epoch_global,n_trivals,n_n_lstm,dropout,bat_size)       
        print('accuracy_avg_1: ',accuracy_avg_1)
        print('accuracy_avg_2: ',accuracy_avg_2)
        print('avg_metrics_prec_recall_F1= ',avg_metrics_prec_recall_F1)
       
    else:        
        vec_mean_metrics = run_ML(model_id,n_steps_out)
        
        mean_all  = np.mean(vec_mean_metrics,axis=0)
        print('vec_mean_metrics',vec_mean_metrics)
        print('_mean_all',mean_all)

main() 