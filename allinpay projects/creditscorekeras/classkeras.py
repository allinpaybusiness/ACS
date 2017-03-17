# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from creditscore.creditscore import CreditScore
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.regularizers import l2
from keras.optimizers import SGD


class CreditScoreKeras(CreditScore):
    
    def dnn_model(self, X_train, y_train, X_test, nepoch, batches):
        #建立DNN模型
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
            
        #训练模型
        model.fit(X_train.values, y_train.values, epochs=nepoch, batch_size=int(X_train.shape[0]/batches)) 
        
        #预测
        probability = model.predict_proba(X_test.values)
        
        return probability
    
    def keras_dnn_trainandtest(self, testsize, cv, feature_sel=None, varthreshold=0, nepoch=10, batches=5, nclusters=10, cmethod=None):

        #分割数据集为训练集和测试集
        data_feature = self.data.ix[:, self.data.columns != 'default']
        data_target = self.data['default']
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        
        #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
        X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, nclusters, cmethod)
 
            
        #训练并预测模型
        probability = self.dnn_model(X_train, y_train, X_test, nepoch, batches)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,0]})      
        
        return predresult
     
    def keras_dnn_trainandtest_kfold(self, nsplit, cv, feature_sel=None, varthreshold=0, nepoch=10, batches=5, nclusters=10, cmethod=None):

        data_feature = self.data.ix[:, self.data.columns != 'default']
        data_target = self.data['default'] 

        #将数据集分割成k个分段分别进行训练和测试，对每个分段，该分段为测试集，其余数据为训练集
        kf = KFold(n_splits=nsplit, shuffle=True)
        predresult = pd.DataFrame()
        for train_index, test_index in kf.split(data_feature):
            X_train, X_test = data_feature.iloc[train_index, ], data_feature.iloc[test_index, ]
            y_train, y_test = data_target.iloc[train_index, ], data_target.iloc[test_index, ]
            
            #如果随机抽样造成train或者test中只有一个分类，跳过此次预测
            if (len(y_train.unique()) == 1) or (len(y_test.unique()) == 1):
                continue
            
            #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
            X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, nclusters, cmethod)
            
            #训练并预测模型
            probability = self.dnn_model(X_train, y_train, X_test, nepoch, batches)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,0]})
            predresult = pd.concat([predresult, temp], ignore_index = True)                    

            
        return predresult
        
        
        
        
       
        
        
        