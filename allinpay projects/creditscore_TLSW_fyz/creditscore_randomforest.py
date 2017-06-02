# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscore_TLSW_fyz.creditscore_tlsw
reload(creditscore_TLSW_fyz.creditscore_tlsw)
from creditscore_TLSW_fyz.creditscore_tlsw import TLSWscoring
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier

class TLSWscoring_randomforest(TLSWscoring):
    
    def RF_trainandtest(self, unionscores, cutscore, testsize, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod):
        
        #分割数据集为训练集和测试集
        if unionscores == True:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'rsk_score'], axis = 1)
        else:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'cst_score',
                                           'cnp_score', 'cnt_score', 'chv_score', 'dsi_score','rsk_score'], axis = 1)
        data_target = (self.data['rsk_score'] < cutscore).astype('int')
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        if testsize == 0:
            X_test, y_test = X_train.head(5), y_train.head(5)
            
        #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
        X_train, X_test = self.binandwoe_traintest_pkl(X_train, y_train, X_test, nclusters, cmethod, self.label)
       
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

        testcolumns = X_test1.columns 
        
        #重采样resampling 解决样本不平衡问题
        X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod) 
            
        #训练并预测随机森林模型
        if rfmethod == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
        elif rfmethod == 'ExtraTrees':
            classifier = ExtraTreesClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
        elif rfmethod == 'GradientBoosting':
            classifier = GradientBoostingClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)

        classifier.fit(X_train1, y_train)  
        probability = classifier.predict_proba(X_test1)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        predresult = pd.concat([predresult, X_test], axis = 1)

        if self.label != None:#label==None 用于建模训练，label！=None用于保存生产模型
            joblib.dump(classifier, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\classifier_" + self.label + '.pkl')
            joblib.dump(testcolumns, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\testcolumns_" + self.label + '.pkl')
        
        
        return predresult
     
    def RF_trainandtest_kfold(self, unionscores, nsplit, cutscore, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod):
        
        if unionscores == True:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'rsk_score'], axis = 1)
        else:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'cst_score',
                                           'cnp_score', 'cnt_score', 'chv_score', 'dsi_score','rsk_score'], axis = 1)
        data_target = (self.data['rsk_score'] < cutscore).astype('int')

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

            #重采样resampling 解决样本不平衡问题
            X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod)
            
            #训练并预测随机森林模型
            if rfmethod == 'RandomForest':
                classifier = RandomForestClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            elif rfmethod == 'ExtraTrees':
                classifier = ExtraTreesClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            elif rfmethod == 'GradientBoosting':
                classifier = GradientBoostingClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)

            classifier.fit(X_train1, y_train)  
            probability = classifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)        

            
        return predresult
        
    def RF_bagging_trainandtest(self, unionscores, cutscore, testsize, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod):
        
        #分割数据集为训练集和测试集
        if unionscores == True:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'rsk_score'], axis = 1)
        else:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'cst_score',
                                           'cnp_score', 'cnt_score', 'chv_score', 'dsi_score','rsk_score'], axis = 1)
        data_target = (self.data['rsk_score'] < cutscore).astype('int')
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        if testsize == 0:
            X_test, y_test = X_train.head(5), y_train.head(5)
            
        #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
        X_train, X_test = self.binandwoe_traintest_pkl(X_train, y_train, X_test, nclusters, cmethod, self.label)
       
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

        testcolumns = X_test1.columns 
        
        #重采样resampling 解决样本不平衡问题
        X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod) 
            
        #训练并预测随机森林模型
        if rfmethod == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
        elif rfmethod == 'ExtraTrees':
            classifier = ExtraTreesClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
        elif rfmethod == 'GradientBoosting':
            classifier = GradientBoostingClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)

        baggingclassifier = BaggingClassifier(classifier, max_samples=0.5) 
        baggingclassifier.fit(X_train1, y_train)  
        probability = baggingclassifier.predict_proba(X_test1)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        predresult = pd.concat([predresult, X_test], axis = 1)

        if self.label != None:#label==None 用于建模训练，label！=None用于保存生产模型
            joblib.dump(baggingclassifier, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\classifier_" + self.label + '.pkl')
            joblib.dump(testcolumns, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\testcolumns_" + self.label + '.pkl')
        
        
        return predresult
     
    def RF_bagging_trainandtest_kfold(self, unionscores, nsplit, cutscore, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod):
        
        if unionscores == True:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'rsk_score'], axis = 1)
        else:
            data_feature = self.data.drop(['name', 'idCard', 'mobileNum', 'cardNum', 'cst_score',
                                           'cnp_score', 'cnt_score', 'chv_score', 'dsi_score','rsk_score'], axis = 1)
        data_target = (self.data['rsk_score'] < cutscore).astype('int')

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

            #重采样resampling 解决样本不平衡问题
            X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod)
            
            #训练并预测随机森林模型
            if rfmethod == 'RandomForest':
                classifier = RandomForestClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            elif rfmethod == 'ExtraTrees':
                classifier = ExtraTreesClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            elif rfmethod == 'GradientBoosting':
                classifier = GradientBoostingClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)

            baggingclassifier = BaggingClassifier(classifier, max_samples=0.5) 
            baggingclassifier.fit(X_train1, y_train)  
            probability = baggingclassifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)        

            
        return predresult        
        
        
       
        
        
        