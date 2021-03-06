# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
import os;
sys.path.append("allinpay projects")
from imp import reload
import creditscore_TLSW_fyz.creditscore_tlsw
reload(creditscore_TLSW_fyz.creditscore_tlsw)
from creditscore_TLSW_fyz.creditscore_tlsw import TLSWscoring
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib



class TLSWscoring_logistic(TLSWscoring):
        
    def logistic_trainandtest(self, unionscores, cutscore, testsize, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, preprocess):

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
        #缺失值处理
        #均值
        #imp  = Imputer(missing_values='NaN', strategy='mean', axis=0)
        #中位数
        #imp  = Imputer(missing_values='NaN', strategy='median', axis=0)
        #最频繁出现的
        #imp  = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        #X_train1 = imp.fit_transform(X_train1)
        #X_test1 = imp.transform(X_test1)
        #数据预处理
        X_train1, X_test1 = self.preprocessData (X_train1, X_test1, preprocess)
            
        #重采样resampling 解决样本不平衡问题
        X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod) 
            
        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train1, y_train)  
        probability = classifier.predict_proba(X_test1)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        predresult = pd.concat([predresult, X_test], axis = 1)
        
        if self.label != None:#label==None 用于建模训练，label！=None用于保存生产模型
            joblib.dump(classifier, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\classifier_" + self.label + '.pkl')
            joblib.dump(testcolumns, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\testcolumns_" + self.label + '.pkl')
            
        return predresult
        
    def logistic_trainandtest_kfold(self, unionscores, nsplit, cutscore, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, preprocess):

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
                       
            #缺失值处理
            #均值
            #imp  = Imputer(missing_values='NaN', strategy='mean', axis=0)
            #中位数
            #imp  = Imputer(missing_values='NaN', strategy='median', axis=0)
            #最频繁出现的
            #imp  = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
            #X_train1 = imp.fit_transform(X_train1)
            #X_test1 = imp.transform(X_test1)
            #数据预处理
            X_train1, X_test1 = self.preprocessData (X_train1, X_test1, preprocess)
            #重采样resampling 解决样本不平衡问题
            X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod)
            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(X_train1, y_train)  
            probability = classifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            temp = pd.concat([temp, data_feature.iloc[test_index, ]], axis = 1)
            predresult = pd.concat([predresult, temp], ignore_index = True)        
            
        return predresult

    def logistic_bagging_trainandtest(self, unionscores, cutscore, testsize, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, preprocess):

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
        #缺失值处理
        #均值
        #imp  = Imputer(missing_values='NaN', strategy='mean', axis=0)
        #中位数
        #imp  = Imputer(missing_values='NaN', strategy='median', axis=0)
        #最频繁出现的
        #imp  = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        #X_train1 = imp.fit_transform(X_train1)
        #X_test1 = imp.transform(X_test1)
        #数据预处理
        X_train1, X_test1 = self.preprocessData (X_train1, X_test1, preprocess)
            
        #重采样resampling 解决样本不平衡问题
        X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod) 
            
        #训练并预测模型
        # 使用类，参数全是默认的
        classifier = BaggingClassifier(LogisticRegression(), max_samples=0.5)
        classifier.fit(X_train1, y_train)  
        probability = classifier.predict_proba(X_test1)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        predresult = pd.concat([predresult, X_test], axis = 1)

        if self.label != None:#label==None 用于建模训练，label！=None用于保存生产模型
            joblib.dump(classifier, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\classifier_" + self.label + '.pkl')
            joblib.dump(testcolumns, "allinpay projects\\creditscore_TLSW_fyz\\pkl\\testcolumns_" + self.label + '.pkl')
        
        return predresult

    def logistic_bagging_trainandtest_kfold(self, unionscores, nsplit, cutscore, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, preprocess):

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
                       
            #缺失值处理
            #均值
            #imp  = Imputer(missing_values='NaN', strategy='mean', axis=0)
            #中位数
            #imp  = Imputer(missing_values='NaN', strategy='median', axis=0)
            #最频繁出现的
            #imp  = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
            #X_train1 = imp.fit_transform(X_train1)
            #X_test1 = imp.transform(X_test1)
            #数据预处理
            X_train1, X_test1 = self.preprocessData (X_train1, X_test1, preprocess)
            #重采样resampling 解决样本不平衡问题
            X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod)
            
            #训练并预测模型
            # 使用类，参数全是默认的
            bagging = BaggingClassifier(LogisticRegression(), max_samples=0.5)
            bagging.fit(X_train1, y_train)  
            probability = bagging.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            temp = pd.concat([temp, data_feature.iloc[test_index, ]], axis = 1)
            predresult = pd.concat([predresult, temp], ignore_index = True)        
            
        return predresult
        

    
         
        

        
    
        
        
        
        
       
        
        
        
