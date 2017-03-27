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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class CreditScoreSVC(CreditScore):

    
    def SVC_trainandtest(self, testsize, cv, feature_sel, varthreshold, nclusters=10, cmethod=None):
        
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
            
        #训练并预测SVC模型
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                 {'kernel': ['sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
        classifier = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5)
        #SVC(kernel=kernel, probability=True)
        classifier.fit(X_train1, y_train)  
        probability = classifier.predict_proba(X_test1)[:,1]
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability})
        
        return predresult

    def SVC_Logistic_trainandtest(self, testsize, cv, feature_sel, varthreshold, nclusters=10, cmethod=None):
        #先用svc过滤，再用logistic评分
        
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
            
        #训练并预测SVC模型
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                 {'kernel': ['sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
        classifier = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5)
        #SVC(kernel=kernel, probability=True)
        classifier.fit(X_train1, y_train)  
        svcpred = classifier.predict(X_test1)

        #训练并预测Logistic模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train1, y_train)  
        #predicted = classifier.predict(X_test)
        probability = classifier.predict_proba(X_test1)[:,1]
        probability[svcpred==1] = 1        
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability})
        
        return predresult
        
    def SVC_trainandtest_kfold(self, nsplit, cv, feature_sel, varthreshold, nclusters=10, cmethod=None):
        
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
            

            #训练并预测随机森林模型
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                     {'kernel': ['sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
            classifier = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5)
            #classifier = SVC(kernel=kernel, probability=True)
            classifier.fit(X_train1, y_train)  
            probability = classifier.predict_proba(X_test1)[:,1]

            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability})
            predresult = pd.concat([predresult, temp], ignore_index = True)        

            
        return predresult
        

        
        
        
       
        
        
        