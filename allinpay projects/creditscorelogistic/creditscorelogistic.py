# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from creditscore.creditscore import CreditScore
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV



class CreditScoreLogistic(CreditScore):
    
    def logistic_trainandtest(self, binn, testsize, cv, feature_sel=None, varthreshold=0, bq=False):
        
        #变量粗分类和woe转化
        datawoe = self.binandwoe(binn, bq)
            
        
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
        #predicted = classifier.predict(X_test)
        probability = classifier.predict_proba(X_test)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        
        return predresult
        
    def logistic_trainandtest_kfold(self, binn, nsplit, cv, feature_sel=None, varthreshold=0, bq=False):
        
        #变量粗分类和woe转化
        
        datawoe = self.binandwoe(binn, bq)
        
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
        
        #将数据集分割成k个分段分别进行训练和测试，对每个分段，该分段为测试集，其余数据为训练集
        kf = KFold(n_splits=nsplit, shuffle=True)
        predresult = pd.DataFrame()
        for train_index, test_index in kf.split(data_feature_sel):
            X_train, X_test = data_feature_sel.iloc[train_index, ], data_feature_sel.iloc[test_index, ]
            y_train, y_test = data_target.iloc[train_index, ], data_target.iloc[test_index, ]
            
            #如果随机抽样造成train或者test中只有一个分类，跳过此次预测
            if (len(y_train.unique()) == 1) or (len(y_test.unique()) == 1):
                continue
            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(X_train, y_train)  
            #predicted = classifier.predict(X_test)
            probability = classifier.predict_proba(X_test)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)        
            
        return predresult
        
    def looplogistic_trainandtest(self, testsize, cv, feature_sel=None, varthreshold=0, bq=False):
        df = pd.DataFrame()
        for i in range (3 , 101):
            predresult = self.logistic_trainandtest(i, testsize, cv, feature_sel, bq=bq)
            auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
            temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks ,'p0=0.5' :metrics_p['accuracy'][5]} ,index=[0])
            df = pd.concat([df, temp], ignore_index = False)
            print('num %s complete' %i)
        time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        df.to_csv(time0+'.csv',index=False,sep=',') 
        
    def looplogistic_trainandtest_kfold(self, nsplit, cv, feature_sel=None, varthreshold=0, bq=False ,op='liblinear'):
         df = pd.DataFrame()
         for i in range (3 , 101):
             predresult = self.logistic_trainandtest_kfold(i, nsplit, cv, feature_sel, bq=bq ,op=op)
             auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
             temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks,'p0=0.5,accuracy' :metrics_p['accuracy'][5]} ,index=[0])
             df = pd.concat([df, temp], ignore_index = True)
             print(' num %s complete' %i)
         time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
         df.to_csv(time0+'-kfold-'+op+'-'+self.dataname+'.csv',index=False,sep=',')  
        
    def looplogistic_trainandtest_kfold_LRCV(self, nsplit, cv, feature_sel=None, varthreshold=0, bq=False ,op='liblinear'):
         df = pd.DataFrame()
         for i in range (3 , 101):
             predresult = self.logistic_trainandtest_kfold_LRCV(i, nsplit, cv, feature_sel, bq=bq ,op=op)
             auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
             temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks,'p0=0.5,accuracy' :metrics_p['accuracy'][5]} ,index=[0])
             df = pd.concat([df, temp], ignore_index = True)
             print(' num %s complete' %i)
         time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
         df.to_csv(time0+'-kfold_LRCV-'+op+'-'+self.dataname+'.csv',index=False,sep=',')  
        
    def logistic_trainandtest_kfold_LRCV(self, binn, nsplit, cv, feature_sel=None, varthreshold=0, bq=False ,op='liblinear'):
        
        #变量粗分类和woe转化
        
        datawoe = self.binandwoe(binn, bq)
        
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
        
        #将数据集分割成k个分段分别进行训练和测试，对每个分段，该分段为测试集，其余数据为训练集
        kf = KFold(n_splits=nsplit, shuffle=True)
        predresult = pd.DataFrame()
        for train_index, test_index in kf.split(data_feature_sel):
            X_train, X_test = data_feature_sel.iloc[train_index, ], data_feature_sel.iloc[test_index, ]
            y_train, y_test = data_target.iloc[train_index, ], data_target.iloc[test_index, ]
            
            #如果随机抽样造成train或者test中只有一个分类，跳过此次预测
            if (len(y_train.unique()) == 1) or (len(y_test.unique()) == 1):
                continue
            
            #训练并预测模型
            classifier = LogisticRegressionCV(cv=nsplit,solver=op)  # 使用类，参数全是默认的
            classifier.fit(X_train, y_train)  
            #predicted = classifier.predict(X_test)
            probability = classifier.predict_proba(X_test)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)
            
        
            
        return predresult
        
    
        
        
        
        
       
        
        
        
