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

        #分割数据集为训练集和测试集
        data_feature = self.data.ix[:, self.data.columns != 'default']
        data_target = self.data['default']
        X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        
        #对训练集做变量粗分类和woe转化，并据此对测试集做粗分类和woe转化
        X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, binn, bq)
            
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
        else:
            X_train1, X_test1 = X_train, X_test        
            
        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train1, y_train)  
        #predicted = classifier.predict(X_test)
        probability = classifier.predict_proba(X_test1)
        
        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        
        return predresult
        
    def logistic_trainandtest_kfold(self, binn, nsplit, cv, feature_sel=None, varthreshold=0, bq=False):

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
            X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, binn, bq)
                    
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
            else:
                X_train1, X_test1 = X_train, X_test          

            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(X_train1, y_train)  
            #predicted = classifier.predict(X_test)
            probability = classifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)        
            
        return predresult

    def logistic_trainandtest_kfold_LRCV(self, binn, nsplit, cv, feature_sel=None, varthreshold=0, bq=False ,op='liblinear'):
        
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
            X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, binn, bq)
                    
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
            else:
                X_train1, X_test1 = X_train, X_test          

            
            #训练并预测模型
            classifier = LogisticRegressionCV(cv=nsplit,solver=op)  # 使用类，参数全是默认的
            classifier.fit(X_train1, y_train)  
            #predicted = classifier.predict(X_test)
            probability = classifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)       
            
        return predresult
        
    def looplogistic_trainandtest(self, testsize, cv, feature_sel=None, varthreshold=0, bq=False):
        df = pd.DataFrame()
        for i in range (3 , 101):#对bin做循环
            #分割train test做测试
            predresult = self.logistic_trainandtest(i, testsize, cv, feature_sel, varthreshold, bq)
            #评估并保存测试结果
            auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
            temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks ,'p0=0.5' :metrics_p['accuracy'][5]} ,index=[0])
            df = pd.concat([df, temp], ignore_index = False)
            print('num %s complete' %i)
        time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        df.to_csv(time0+'.csv',index=False,sep=',') 
        
    def looplogistic_trainandtest_kfold(self, nsplit, cv, feature_sel=None, varthreshold=0, bq=False):
         df = pd.DataFrame()
         for i in range (3 , 101):#对bin做循环
             #做cross validation测试
             predresult = self.logistic_trainandtest_kfold(i, nsplit, cv, feature_sel, varthreshold, bq)
             #评估并保存测试结果
             auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
             temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks,'p0=0.5,accuracy' :metrics_p['accuracy'][5]} ,index=[0])
             df = pd.concat([df, temp], ignore_index = True)
             print(' num %s complete' %i)
         time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
         df.to_csv(time0+'-kfold-'+'-'+self.dataname+'.csv',index=False,sep=',')  
        
    def looplogistic_trainandtest_kfold_LRCV(self, nsplit, cv, feature_sel=None, varthreshold=0, bq=False ,op='liblinear'):
         df = pd.DataFrame()
         for i in range (3 , 101):#对bin做循环
             #做cross validation cv测试
             predresult = self.logistic_trainandtest_kfold_LRCV(i, nsplit, cv, feature_sel, varthreshold, bq=bq ,op=op)
             #评估并保存测试结果
             auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
             temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks,'p0=0.5,accuracy' :metrics_p['accuracy'][5]} ,index=[0])
             df = pd.concat([df, temp], ignore_index = True)
             print(' num %s complete' %i)
         time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
         df.to_csv(time0+'-kfold_LRCV-'+op+'-'+self.dataname+'.csv',index=False,sep=',')
    
         
    def logistic_trainandtest_kfold_nowoe(self, nsplit, cv, feature_sel=None, varthreshold=0):

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
            X_train, X_test = self.binandwoe_traintest(X_train, y_train, X_test, 10 ,True)
                    
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
            else:
                X_train1, X_test1 = X_train, X_test          

            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(X_train1, y_train)  
            #predicted = classifier.predict(X_test)
            probability = classifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)        
            
        return predresult
        

        
    
        
        
        
        
       
        
        
        
