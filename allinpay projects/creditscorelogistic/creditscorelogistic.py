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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV


class CreditScoreLogistic(CreditScore):
    
    def logistic_trainandtest(self, binn, testsize, cv, feature_sel=None, varthreshold=0):
        
        #变量粗分类和woe转化
        datawoe = self.binandwoe(binn)
        
        #cross validation 测试
        data_feature = datawoe.ix[:, datawoe.columns != 'default']
        data_target = datawoe['default']

        
        #变量筛选, sklearn.feature_selection中的方法
        if feature_sel == "VarianceThreshold":
            selector = VarianceThreshold(threshold = varthreshold)
            data_feature_sel = pd.DataFrame(selector.fit_transform(data_feature))
            data_feature_sel.columns = data_feature.columns[selector.get_support(True)]
        elif feature_sel == "RFECV":
            estimator = LogisticRegression()
            selector = RFECV(estimator, step=1, cv=cv)
            data_feature_sel = pd.DataFrame(selector.fit_transform(data_feature, data_target))
            data_feature_sel.columns = data_feature.columns[selector.get_support(True)]
        else:
            data_feature_sel = data_feature
        
        #分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(data_feature_sel, data_target, test_size=testsize, random_state=0)

        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train, y_train)  
        predicted = classifier.predict(X_test)
        probability = classifier.predict_proba(X_test)
        
        predresult = pd.DataFrame({'target' : y_test, 'predicted' : predicted, 'probability' : probability[:,1]})
        
        return predresult
     
    '''    
    def logistic_crossvalidation(self, binn, testsize, cv, feature_sel=None, varthreshold=0):
        
        datawoe = self.binandwoe(binn)
        
        #cross validation 测试
        data_feature = datawoe.ix[:, datawoe.columns != 'default']
        data_target = datawoe['default']

        
        #变量筛选, sklearn.feature_selection中的方法
        if feature_sel == "VarianceThreshold":
            selector = VarianceThreshold(threshold = varthreshold)
            data_feature_sel = pd.DataFrame(selector.fit_transform(data_feature))
            data_feature_sel.columns = data_feature.columns[selector.get_support(True)]
        elif feature_sel == "RFECV":
            estimator = LogisticRegression()
            selector = RFECV(estimator, step=1, cv=cv)
            data_feature_sel = pd.DataFrame(selector.fit_transform(data_feature, data_target))
            data_feature_sel.columns = data_feature.columns[selector.get_support(True)]
        else:
            data_feature_sel = data_feature
        
        """
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train, y_train)   # train test 样本外测试
        classifier.score(X_test, y_test)
        """
        clf = LogisticRegression()
        #score = cross_val_score(clf, data_feature_sel, data_target, cv=cv)   
        predicted = cross_val_predict(clf, data_feature_sel, data_target, cv=cv)
        
        predtable = pd.DataFrame({'target' : data_target, 'predicted' : predicted})
        
        return predtable
   '''
       
        
        
        