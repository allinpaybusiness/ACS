# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

class CreditClassifier:
    
    def __init__(self, dataname):
        #dataname指定导入那个数据集
        self.dataname = dataname
        
        if self.dataname == 'german':
            self.data = pd.read_table('raw data\\German data\\german.data.txt',header=None,delim_whitespace=True)
            #重新命名特征变量A1A2A3...和违约变量default
            names = ['A1']
            for i in range(1,self.data.shape[1] - 1):
                names.append('A' + str(i+1))
            names.append('default')
            self.data.columns = names
            #german数据中1=good 2=bad 转换成0=good 1=bad
            self.data.default = self.data.default.replace({1:0, 2:1})

        if self.dataname == 'HMEQ':
            self.data = pd.read_csv('raw data\\credit scoring\\HMEQ.csv')
            self.data = self.data.rename(columns = {'BAD':'default'})
            
            self.data['MORTDUE'] = pd.to_numeric(self.data['MORTDUE'], errors='coerce')
            self.data['MORTDUE'] = self.data['MORTDUE'].fillna(self.data['MORTDUE'].mean())
            self.data['VALUE'] = pd.to_numeric(self.data['VALUE'], errors='coerce')
            self.data['VALUE'] = self.data['VALUE'].fillna(self.data['VALUE'].mean())
            self.data['REASON'] = self.data['REASON'].fillna("NA")
            self.data['JOB'] = self.data['JOB'].fillna("NA")
            self.data['YOJ'] = pd.to_numeric(self.data['YOJ'], errors='coerce')
            self.data['YOJ'] = self.data['YOJ'].fillna(self.data['YOJ'].mean())
            self.data['DEROG'] = pd.to_numeric(self.data['DEROG'], errors='coerce')
            self.data['DEROG'] = self.data['DEROG'].fillna(self.data['DEROG'].mean())
            self.data['DELINQ'] = pd.to_numeric(self.data['DELINQ'], errors='coerce')
            self.data['DELINQ'] = self.data['DELINQ'].fillna(self.data['DELINQ'].mean())            
            self.data['CLAGE'] = pd.to_numeric(self.data['CLAGE'], errors='coerce')
            self.data['CLAGE'] = self.data['CLAGE'].fillna(self.data['CLAGE'].mean())
            self.data['NINQ'] = pd.to_numeric(self.data['NINQ'], errors='coerce')
            self.data['NINQ'] = self.data['NINQ'].fillna(self.data['NINQ'].mean())
            self.data['CLNO'] = pd.to_numeric(self.data['CLNO'], errors='coerce')
            self.data['CLNO'] = self.data['DEROG'].fillna(self.data['CLNO'].mean())            
            self.data['DEBTINC'] = pd.to_numeric(self.data['DEBTINC'], errors='coerce')
            self.data['DEBTINC'] = self.data['DEBTINC'].fillna(self.data['DEBTINC'].mean())
            
    def binandwoe(self, binn):
        #进行粗分类和woe转换
        datawoe = self.data.copy()
        
        for col in datawoe.columns:
            
            if col == 'default':
                continue
            
            #首先判断是否是名义变量
            if datawoe[col].dtype == 'O':
                for cat in datawoe[col].unique():
                    #计算单个分类的woe
                    nob = max(1, sum((datawoe.default == 1) & (datawoe[col] == cat)))
                    tnob = sum(datawoe.default == 1)
                    nog = max(1, sum((datawoe.default == 0) & (datawoe[col] == cat)))
                    tnog = sum(datawoe.default == 0)
                    woei = np.log((nob/tnob)/(nog/tnog))
                    datawoe[col] = datawoe[col].replace({cat:woei})            
            else:
                #对连续特征粗分类
                minvalue = datawoe[col].min()
                maxvalue = datawoe[col].max()
                breakpoints = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/binn)
                breakpoints = np.append(breakpoints, maxvalue)
                labels = np.arange(len(breakpoints) - 1)
                datawoe[col] = pd.cut(datawoe[col],bins=breakpoints,right=True,labels=labels,include_lowest=True)
                datawoe[col] = datawoe[col].astype('int64')
                
                for cat in datawoe[col].unique():
                    #计算单个分类的woe  
                    nob = max(1, sum((datawoe.default == 1) & (datawoe[col] == cat)))
                    tnob = sum(datawoe.default == 1)
                    nog = max(1, sum((datawoe.default == 0) & (datawoe[col] == cat)))
                    tnog = sum(datawoe.default == 0)
                    woei = np.log((nob/tnob)/(nog/tnog))
                    datawoe[col] = datawoe[col].replace({cat:woei}) 
                    
        return datawoe
            
    def categoricalwoe(self):
        #进行粗分类和woe转换
        datawoe = self.data.copy()
        
        for col in datawoe.columns:
            
            if col == 'default':
                continue
            
            #首先判断是否是名义变量
            if datawoe[col].dtype == 'O':
                for cat in datawoe[col].unique():
                    #计算单个分类的woe
                    nob = max(1, sum((datawoe.default == 1) & (datawoe[col] == cat)))
                    tnob = sum(datawoe.default == 1)
                    nog = max(1, sum((datawoe.default == 0) & (datawoe[col] == cat)))
                    tnog = sum(datawoe.default == 0)
                    woei = np.log((nob/tnob)/(nog/tnog))
                    datawoe[col] = datawoe[col].replace({cat:woei})            
                    
        return datawoe
            
            
    def dataencoder(self):
        
        data = self.data
        
        #引入哑变量
        data_feature = data.ix[:, data.columns != 'default']

        
        data_feature0 = data_feature.ix[:, data_feature.dtypes!='object']
        data_feature1 = pd.DataFrame()
        for col in data_feature.columns:
            if data_feature[col].dtype == 'O':
                le = preprocessing.LabelEncoder()
                temp = pd.DataFrame(le.fit_transform(data_feature[col]), columns=[col])
                data_feature1 = pd.concat([data_feature1, temp], axis=1)
                
        enc = preprocessing.OneHotEncoder()
        data_feature1_enc = pd.DataFrame(enc.fit_transform(data_feature1).toarray())
        data_feature_enc = pd.concat([data_feature0, data_feature1_enc], axis=1)    
        
        return(data_feature_enc)
       
    def modelmetrics_binary(self, predtable):
        
        
        scores = metrics.accuracy_score(predtable['target'], predtable['predicted'])          
        print('cross_validation scores: %s' %scores)         
        
        confusion_matrix = pd.DataFrame(metrics.confusion_matrix(predtable['target'], predtable['predicted']), index=['real_negtive', 'real_postive'], columns=['pred_negtive', 'pred_postive'])  
        confusion_matrix_prob = confusion_matrix.copy()
        confusion_matrix_prob.iloc[:, 0] = confusion_matrix_prob.iloc[:, 0] / confusion_matrix_prob.iloc[:, 0].sum()
        confusion_matrix_prob.iloc[:, 1] = confusion_matrix_prob.iloc[:, 1] / confusion_matrix_prob.iloc[:, 1].sum()

        print(confusion_matrix)     
        print(confusion_matrix_prob) 
        
        precision = metrics.precision_score(predtable['target'], predtable['predicted'])
        recall = metrics.recall_score(predtable['target'], predtable['predicted'])
        print('precision scores: %s' %precision)
        print('recall scores: %s' %recall)        
        
    def modelmetrics_scores(self, predtable):
        pass
        
        