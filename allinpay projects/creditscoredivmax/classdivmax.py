# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:56:14 2017

@author: yesf
"""

##############################################################################
#为方便与logistic模型结果的比对，评分卡为“颠倒型”，也就是分数越高，违约可能越大
#散度最大化的基本想法是将评分卡设置为特征的线性组合后，最大化好人群和坏人群的散度
#好人群和坏人群的散度定义为：
# 目标函数：max [f(s,G)-f(s,B)]ln[f(s,G)/f(s,B)]ds (Divergence)
# 其简化形式为正态分布化后的散度(Divergence_Normal)，或马氏距离(Mahal_Dist)
# 评分卡形式依然为线性组合：sum(C_j*X_ij)
##############################################################################


import sys;
import os;
sys.path.append("allinpay projects/creditscoredivmax")
from imp import reload
import creditscore
reload(creditscore)
from creditscore import CreditScore
import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest



class CreditScoreDivergenceMax(CreditScore):
    
    def DivMax_trainandtest(self, testsize, cv, feature_sel, varthreshold, nclusters, \
                             cmethod, resmethod):

        #分割数据集为训练集和测试集
        data_feature = self.data.ix[:, self.data.columns != 'default']
        data_target = self.data['default']
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        
        
        #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
        X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, nclusters, cmethod)
            
        #在train中做变量筛选, sklearn.feature_selection中的方法
        if feature_sel == "VarianceThreshold":
            selector = VarianceThreshold(threshold = varthreshold)
            X_train1 = pd.DataFrame(selector.fit_transform(X_train))
            X_train1.columns = X_train.columns[selector.get_support(True)]
            X_test1 = X_test[X_train1.columns]
        elif feature_sel == "RFECV":
            estimator = LogisticRegression()
            selector = RFECV(estimator, step=1, cv=cv)
            X_train1 = pd.DataFrame(selector.fit_transform(X_train, y_train))
            X_train1.columns = X_train.columns[selector.get_support(True)]
            X_test1 = X_test[X_train1.columns]
        elif feature_sel == "SelectFromModel":
            estimator = LogisticRegression()
            selector = SelectFromModel(estimator)
            X_train1 = pd.DataFrame(selector.fit_transform(X_train, y_train))
            X_train1.columns = X_train.columns[selector.get_support(True)]
            X_test1 = X_test[X_train1.columns]
        elif feature_sel == "SelectKBest":
            selector = SelectKBest()
            X_train1 = pd.DataFrame(selector.fit_transform(X_train, y_train))
            X_train1.columns = X_train.columns[selector.get_support(True)]
            X_test1 = X_test[X_train1.columns]
        else:
            X_train1, X_test1 = X_train, X_test        
            
        #重采样resampling 解决样本不平衡问题
        X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod)             
        
        X_train_good = []
        X_train_bad = []
        for i in range(X_train1.shape[0]):
            if y_train.iloc[i] == 0:
                X_train_good.append(list(X_train1.iloc[i,:]))
            elif y_train.iloc[i] == 1:
                X_train_bad.append(list(X_train1.iloc[i,:]))
            else:
                print('error')
                
        # 目标函数
        fun = lambda x: -Divergence_Normal(np.dot(X_train_good,x), np.dot(X_train_bad,x))
        init = [0.5]*X_train1.shape[1]

        res = minimize(fun, init, method = "CG",  tol = 1e-9, options={'gtol': 1e-6, 'disp': True})

        predcoeff = res.x[:X_train1.shape[1]]

        for i in range(0,len(predcoeff)):
            print(i+1, predcoeff[i])

        score = np.dot(X_test1,predcoeff)
        
        # 需要找到一个得分分布到概率分布的转换：下面为简单线性转换
        probability = (score-score.min())/(score.max()-score.min())

        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability})
        
        return predresult


def Divergence(list1, list2):
# This function calculates the divergence of good points (in list1) and bad points (in list2)   

        max_list = max(max(list1),max(list2))
        min_list = min(min(list1),min(list2))
        sample_size = len(list1) + len(list2)
        num_cat = max(int(sample_size / 20),1)


        breakpoints = np.arange(min_list, max_list, (max_list-min_list)/num_cat) 
        breakpoints = np.append(breakpoints, max_list)
        
        
        labels = np.arange(len(breakpoints) - 1)
        datacut1 = pd.cut(list1,bins=breakpoints,right=True,labels=labels,include_lowest=True)
        datacut2 = pd.cut(list2,bins=breakpoints,right=True,labels=labels,include_lowest=True)
        
        div = 0
        for cat in labels:
            dist1 = (datacut1 == cat).sum() / len(list1)
            dist2 = (datacut2 == cat).sum() / len(list2)
            if dist1 == 0:
                dist1 = 0.0001
            if dist2 == 0:
                dist2 = 0.0001
            div = div + (dist1 - dist2) * np.log (dist1/dist2)
        
        return div


def Divergence_Normal(list1, list2):
# This function calculates the divergence of good points (in list1) and bad points (in list2)   
# by assuming the good and bad points follow the gaussian distribution    

        mean1 = np.mean(list1)
        mean2 = np.mean(list2)
        stdev1 = np.std(list1)
        stdev2 = np.std(list2)

        dist = ((stdev1**2 + stdev2**2)*(mean1 - mean2)**2 + (stdev1**2 - stdev2**2)**2) / (2 * stdev1**2 * stdev2**2)
      
        return dist
    

def Mahal_Dist(list1, list2):
# This function calculates the mahal distance of good points (in list1) and bad points (in list2)   
# by assuming the good and bad points follow the gaussian distribution and same stdev  

        mean1 = np.mean(list1)
        mean2 = np.mean(list2)
        stdev1 = np.std(list1)
        stdev2 = np.std(list2)
        stdev = (stdev1 + stdev2)/2

        dist = abs(mean1 - mean2) / stdev
      
        return dist