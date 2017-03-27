# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
import os;
sys.path.append("allinpay projects")
from creditscore.creditscore import CreditScore
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

class CreditScoreMLP(CreditScore):
    
    def MLP_trainandtest(self, testsize, cv, feature_sel, varthreshold, activation,solver, alpha,  max_iter =1000,nclusters=10, cmethod=None, *hidden_layer_sizes):
        
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
            
        #训练并预测模型

        classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,solver=solver,alpha=alpha, max_iter =1000)  # 使用类，参数全是默认的

        #为避免单次神经网络训练不收敛的情况，反复训练10次，最终预测概率为10次的平均值
        probability = 0
        for i in range(10):
            #训练模型
            classifier.fit(X_train1, y_train)  
            #预测概率
            probability += classifier.predict_proba(X_test1)[:,1]
            print("probability == %s" % (classifier.predict_proba(X_test1)[:,1]))
        probability = probability / 10
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability})
        
        return predresult
     
    def MLP_trainandtest_kfold(self, nsplit, cv, feature_sel, varthreshold, activation,solver, alpha,  max_iter =1000,nclusters=10, cmethod=None, *hidden_layer_sizes):
        
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
            
            #训练并预测模型

            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,solver=solver, alpha=alpha,max_iter =max_iter)  # 使用类，参数全是默认的

            #为避免单次神经网络训练不收敛的情况，反复训练10次，最终预测概率为10次的平均值
            probability = 0
            for i in range(10):
                #训练模型
                classifier.fit(X_train1, y_train)  
                #预测概率
                probability += classifier.predict_proba(X_test1)[:,1]
                print("probability == %s" % (classifier.predict_proba(X_test1)[:,1]))
            probability = probability / 10
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability})
            predresult = pd.concat([predresult, temp], ignore_index = True)        

            
        return predresult
       
    def loopMLP_trainandtest(self, testsize, cv, feature_sel, varthreshold, activation, solver,alpha, max_iter =1000, nclusters=10, cmethod=None):
        df = pd.DataFrame()
        for i in range (3 , 101,3):#对神经元做循环
            hidden_layer_sizes = (i,)
            #分割train test做测试
            predresult = self.MLP_trainandtest(testsize, cv, feature_sel, varthreshold, activation,solver ,alpha, max_iter,nclusters, cmethod, *hidden_layer_sizes)
            #评估并保存测试结果
            auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
            temp = pd.DataFrame({'hidden_first_layer' : i, 'auc_value' : auc ,'ks_value' :ks ,'p0=0.5' :metrics_p['accuracy'][5]} ,index=[0])
            df = pd.concat([df, temp], ignore_index = False)
            print('num %s complete' %i)
        time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        exist = os.path.exists('d:/ACS_CSVS')
        if exist:
            df.to_csv('d:/ACS_CSVS/'+time0+'_MLP.csv',index=False,sep=',') 
        else:
            os.makedirs('d:/ACS_CSVS/')
            df.to_csv('d:/ACS_CSVS/'+time0+'_MLP.csv',index=False,sep=',') 
            
    def loopMLP_trainandtest_kfold(self, testsize, cv, feature_sel, varthreshold, activation, solver,alpha, max_iter =1000, nclusters=10, cmethod=None):
        df = pd.DataFrame()
        for i in range (3 , 101,3):#对神经元做循环
            hidden_layer_sizes = (i,)
            #分割train test做测试
            predresult = self.MLP_trainandtest_kfold(testsize, cv, feature_sel, varthreshold, activation,solver ,alpha, max_iter,nclusters, cmethod, *hidden_layer_sizes)
            #评估并保存测试结果
            auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
            temp = pd.DataFrame({'hidden_first_layer' : i, 'auc_value' : auc ,'ks_value' :ks ,'p0=0.5' :metrics_p['accuracy'][5]} ,index=[0])
            df = pd.concat([df, temp], ignore_index = False)
            print('num %s complete' %i)
        time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        exist = os.path.exists('d:/ACS_CSVS')
        if exist:
            df.to_csv('d:/ACS_CSVS/'+time0+'_MLP.csv',index=False,sep=',') 
        else:
            os.makedirs('d:/ACS_CSVS/')
            df.to_csv('d:/ACS_CSVS/'+time0+'_MLP.csv',index=False,sep=',') 
        
        
        
       
        
        
        