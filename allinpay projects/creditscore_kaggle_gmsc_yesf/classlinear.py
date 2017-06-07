# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
import os;
sys.path.append("allinpay projects")
from imp import reload
import creditscore
reload(creditscore)
from creditscore import CreditScore
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegressionCV
#from sklearn.model_selection import KFold
#from sklearn.feature_selection import VarianceThreshold
#from sklearn.feature_selection import RFECV
#from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA


class CreditScoreLinear(CreditScore):
    
    def linear_trainandtest(self, testsize):
                            #cv, feature_sel, varthreshold, nclusters, cmethod, resmethod):

        #分割数据集为训练集和测试集
        
        data_feature = self.data.ix[:, self.data.columns != '风险得分']

        data_target = self.data['风险得分']
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)

#        for col in X_train.columns:
#            datatemp = X_train[col].copy()
#            min_temp = datatemp.min()
#            max_temp = datatemp.max()
#            if min_temp != max_temp:
#                X_train[col] = (datatemp - min_temp) / (max_temp - min_temp)

        for col in X_train.columns:
            datatemp = X_train[col].copy()

            stdev = np.std(datatemp)
            if stdev != 0:
                X_train[col] = datatemp / stdev

        datatemp = y_train.copy()
#        min_temp = datatemp.min()
#        max_temp = datatemp.max()
#        if min_temp != max_temp:
#            y_train = (datatemp - min_temp) / (max_temp - min_temp)
        stdev = np.std(datatemp)
        if stdev != 0:
            y_train = datatemp / stdev

        
        #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
#        X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, nclusters, cmethod)
        print('data cleaning finished')
    
        #训练并预测模型
        classifier = LinearRegression(fit_intercept=False)  # 使用类，参数全是默认的
        classifier.fit(X_train, y_train)  
        #predicted = classifier.predict(X_test)
#        probability = classifier.predict_proba(X_test1)
        
        print('regression finished')
#        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})        

        return X_train, y_train, classifier.coef_, classifier.intercept_
#        return X_train, y_train

    def linear_pca(self):
        
        exclusion = pd.read_csv('..\\UnionPay2exclude.csv', encoding="gbk")
        var_exclude = list(exclusion.columns)
        for col in var_exclude: 
            del self.data[col]
        
        data_feature = self.data.ix[:, self.data.columns != '风险得分']
        data_target = (self.data['风险得分']/10).astype(int)

        X_train = data_feature.copy()
                
#        for col in X_train.columns:
#            meantemp = np.mean(X_train[col])
#            X_train[col] = X_train[col] - meantemp
#            min_temp = X_train[col].min()
#            max_temp = X_train[col].max()
##            if min_temp != max_temp:
#                power = np.log10(max_temp).astype(int)+1
#                X_train[col] = X_train[col]/(10**power)
                
#                X_train[col] = (datatemp - min_temp) / (max_temp - min_temp)


        for col in X_train.columns:
            datatemp = X_train[col].copy()

            stdev = np.std(datatemp)
            if stdev != 0:
                X_train[col] = datatemp / stdev
             
        matrix = np.dot(np.transpose(X_train), X_train)#/X_train.shape[0]
        eigenvalues, eigenfunctions = np.linalg.eig(matrix)
        
        
        return X_train, matrix, eigenvalues, eigenfunctions


    def linear_pca_standard(self):
        
        exclusion = pd.read_csv('..\\UnionPay2exclude.csv', encoding="gbk")
        var_exclude = list(exclusion.columns)
        for col in var_exclude: 
            del self.data[col]

        data_feature = self.data.ix[:, self.data.columns != '风险得分']
        data_target = (self.data['风险得分']/10).astype(int)

        X_train = data_feature.copy()

#        for col in X_train.columns:
#            datatemp = X_train[col].copy()
#            min_temp = datatemp.min()
#            max_temp = datatemp.max()
#            if min_temp != max_temp:
#                X_train[col] = (datatemp - min_temp) / (max_temp - min_temp)

        for col in X_train.columns:
            datatemp = X_train[col].copy()

            stdev = np.std(datatemp)
            if stdev != 0:
                X_train[col] = datatemp / stdev
        
        classifier = PCA(n_components = 100, copy = False, whiten = True)
           
        decomposition = classifier.fit_transform(X_train)
        
        explained = classifier.explained_variance_ratio_
        
        return X_train, decomposition, explained


        
  
        
