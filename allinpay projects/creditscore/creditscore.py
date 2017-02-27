# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn import metrics
from sklearn import preprocessing

class CreditScore:
    
    def __init__(self, dataname):
        #dataname指定导入那个数据集
        self.dataname = dataname
        
        #读取数据集
        if self.dataname == 'german':
            self.data = pd.read_table('raw data\\credit scoring\\german.data.txt',header=None,delim_whitespace=True)
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
            
    def binandwoe(self, binn, bq):
        #进行粗分类和woe转换
        #进行粗分类（bin）时，bq=True对连续变量等分位数分段，bp=False对连续变量等宽分段
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
                if bq == True:
                    breakpoints = np.unique(np.percentile(datawoe[col],range(0,110,10)))
                    if len(breakpoints) == 2:
                        breakpoints = np.array([breakpoints[0], np.mean(breakpoints), breakpoints[1]])
                else:
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
       
    def modelmetrics_binary(self, predresult):
        
        #准确率
        scores = metrics.accuracy_score(predresult['target'], predresult['predicted'])          
        print('cross_validation scores: %s' %scores)         
        
        #混合概率矩阵
        confusion_matrix = pd.DataFrame(metrics.confusion_matrix(predresult['target'], predresult['predicted']), index=['real_negtive', 'real_postive'], columns=['pred_negtive', 'pred_postive'])  
        confusion_matrix_prob = confusion_matrix.copy()
        confusion_matrix_prob.iloc[:, 0] = confusion_matrix_prob.iloc[:, 0] / confusion_matrix_prob.iloc[:, 0].sum()
        confusion_matrix_prob.iloc[:, 1] = confusion_matrix_prob.iloc[:, 1] / confusion_matrix_prob.iloc[:, 1].sum()

        print(confusion_matrix)     
        print(confusion_matrix_prob) 
        
        #精确度和召回率
        precision = metrics.precision_score(predresult['target'], predresult['predicted'])
        recall = metrics.recall_score(predresult['target'], predresult['predicted'])
        print('precision scores: %s' %precision)
        print('recall scores: %s' %recall)        
        
    def modelmetrics_scores(self, predresult):
        
        ###AUC KS值
        auc = metrics.roc_auc_score(predresult.target, predresult.probability)
        print('AUC: %s' %auc)
                
        
        ##### KS值
        G = predresult.ix[predresult.target == 0, 'probability']
        B = predresult.ix[predresult.target == 1, 'probability']
        ks,d = ss.ks_2samp(G,B)
        print('ks: %s  d:%s' %(ks,d))
        
        ###在某个概率分界值p下，模型预测的各项准确率
        metrics_p = pd.DataFrame()
        for p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            predresult['predicted'] = (predresult.probability > p).astype(int)
            pred_accuracy = sum(predresult.predicted == predresult.target)/predresult.shape[0]
    
            confusion_matrix = pd.DataFrame(metrics.confusion_matrix(predresult['target'], predresult['predicted']), index=['real_negtive', 'real_postive'], columns=['pred_negtive', 'pred_postive'])  
            confusion_matrix_prob = confusion_matrix.copy()
            confusion_matrix_prob.iloc[:, 0] = confusion_matrix_prob.iloc[:, 0] / confusion_matrix_prob.iloc[:, 0].sum()
            confusion_matrix_prob.iloc[:, 1] = confusion_matrix_prob.iloc[:, 1] / confusion_matrix_prob.iloc[:, 1].sum()
    
            precision = metrics.precision_score(predresult['target'], predresult['predicted'])
            recall = metrics.recall_score(predresult['target'], predresult['predicted'])
            pass_rate = sum(predresult.predicted == 0)/predresult.shape[0]

            temp = pd.DataFrame({'p0': p, 'accuracy': pred_accuracy, 'precision': precision,
                                 'recall': recall, 'pass_rate': pass_rate, 'FalseNegative': confusion_matrix_prob.iloc[1, 0]}, index=[0])
            temp = temp[['p0', 'accuracy', 'precision', 'recall', 'pass_rate', 'FalseNegative']]
            metrics_p = pd.concat([metrics_p, temp], ignore_index = True)
            
        print(metrics_p)
        
        ###画出ROC曲线
        fpr, tpr, _ = metrics.roc_curve(predresult.target, predresult.probability)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
        ###画出KS曲线
        data1 = np.sort(G)
        data2 = np.sort(B)
        n1 = data1.shape[0]
        n2 = data2.shape[0]
        data_all = np.sort(np.concatenate([data1, data2]))
          
        cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0*n1)
        cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0*n2)
        plt.figure()
    
        plt.plot(data_all,cdf1, color='darkorange',lw=2)
        plt.plot(data_all,cdf2, color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('VALUE')
        plt.ylabel('STATS')
        plt.title('KS-CURVE characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        
        
        
        