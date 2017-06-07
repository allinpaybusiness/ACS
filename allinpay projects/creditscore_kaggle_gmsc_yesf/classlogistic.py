# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
import os;
sys.path.append("allinpay projects")
from imp import reload
import creditscore
reload(creditscore)
from creditscore import CreditScore
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA


class CreditScoreLogistic(CreditScore):
    
    def logistic_trainandtest(self, testsize):#, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod):

        
#        inclusion = pd.read_csv('..\\UnionPay2include.csv', encoding="gbk")
        
        
        #分割数据集为训练集和测试集
#        data_feature = self.data.ix[:, [self.data.columns == col for col in inclusion]]
        data_feature = self.data.ix[:, 2:]
        data_target = (self.data['风险得分']/10).astype(int)
#        X_train, Y_train = data_feature, data_target
        X_train, X_test, Y_train, Y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)
        for col in X_train:
            stdev = np.std(X_train[col])
            if stdev != 0:    
                X_train[col] = X_train[col] / stdev
                X_test[col] = X_test[col] / stdev
       

        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train, Y_train)  
        #predicted = classifier.predict(X_test)
        probability = classifier.predict(X_train)
                
#        predresult = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
        return Y_train, probability
        
#        return predresult
        
    def logistic_trainandtest_kfold(self, nsplit, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod):

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

            #重采样resampling 解决样本不平衡问题
            X_train1, y_train = self.imbalanceddata (X_train1, y_train, resmethod)
            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(X_train1, y_train)  
            #predicted = classifier.predict(X_test)
            probability = classifier.predict_proba(X_test1)
            
            temp = pd.DataFrame({'target' : y_test, 'probability' : probability[:,1]})
            predresult = pd.concat([predresult, temp], ignore_index = True)        
            
        return predresult

    def logistic_trainandtest_kfold_LRCV(self, nsplit, cv, feature_sel=None, varthreshold=0, op='liblinear', nclusters=10, cmethod=None):
        
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
        
    def looplogistic_trainandtest(self, testsize, cv, feature_sel=None, varthreshold=0,cmethod=None ):
        df = pd.DataFrame()
        for i in range (3 , 101):#对bin或者ncluster做循环
            #分割train test做测试
            predresult = self.logistic_trainandtest(i, testsize, cv, feature_sel, varthreshold,nclusters=i,cmethod=cmethod)
            #评估并保存测试结果
            auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
            temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks ,'p0=0.5' :metrics_p['accuracy'][5]} ,index=[0])
            df = pd.concat([df, temp], ignore_index = False)
            print('num %s complete' %i)
        time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        exist = os.path.exists('d:/ACS_CSVS')
        if exist:
            df.to_csv('d:/ACS_CSVS/'+time0+'.csv',index=False,sep=',') 
        else:
            os.makedirs('d:/ACS_CSVS/')
            df.to_csv('d:/ACS_CSVS/'+time0+'.csv',index=False,sep=',') 
        
    def looplogistic_trainandtest_kfold(self, nsplit, cv, feature_sel=None, varthreshold=0,cmethod=None):
         df = pd.DataFrame()
         for i in range (3 , 101):#对bin或者ncluster做循环
             #做cross validation测试
             predresult = self.logistic_trainandtest_kfold(i, nsplit, cv, feature_sel, varthreshold,nclusters =i,cmethod=cmethod)
             #评估并保存测试结果
             auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
             temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks,'p0=0.5,accuracy' :metrics_p['accuracy'][5]} ,index=[0])
             df = pd.concat([df, temp], ignore_index = True)
             print(' num %s complete' %i)
         time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
         exist = os.path.exists('d:/ACS_CSVS')
         if exist:
            if cmethod != None:
                df.to_csv('d:/ACS_CSVS/'+time0+'-kfold-'+'-'+self.dataname+'-'+cmethod+'.csv',index=False,sep=',')
            else:
                df.to_csv('d:/ACS_CSVS/'+time0+'-kfold-'+'-'+self.dataname+'.csv',index=False,sep=',')
         else:
            os.makedirs('d:/ACS_CSVS/')
            if cmethod != None:
                df.to_csv('d:/ACS_CSVS/'+time0+'-kfold-'+'-'+self.dataname+'-'+cmethod+'.csv',index=False,sep=',') 
            else:
                df.to_csv('d:/ACS_CSVS/'+time0+'-kfold-'+'-'+self.dataname+'.csv',index=False,sep=',')
        
    def looplogistic_trainandtest_kfold_LRCV(self, nsplit, cv, feature_sel=None, varthreshold=0,op='liblinear',cmethod=None):
         df = pd.DataFrame()
         for i in range (3 , 101):#对bin做循环
             #做cross validation cv测试
             predresult = self.logistic_trainandtest_kfold_LRCV(nsplit, cv, feature_sel, varthreshold ,op=op,nclusters=i)
             #评估并保存测试结果
             auc, ks, metrics_p = self.loopmodelmetrics_scores(predresult)
             temp = pd.DataFrame({'bin' : i, 'auc_value' : auc ,'ks_value' :ks,'p0=0.5,accuracy' :metrics_p['accuracy'][5]} ,index=[0])
             df = pd.concat([df, temp], ignore_index = True)
             print(' num %s complete' %i)
         time0 = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
         exist = os.path.exists('d:/ACS_CSVS')
         if exist:
            df.to_csv('d:/ACS_CSVS/'+time0+'-kfold_LRCV-'+op+'-'+self.dataname+'.csv',index=False,sep=',') 
         else:
            os.makedirs('d:/ACS_CSVS/')
            df.to_csv('d:/ACS_CSVS/'+time0+'-kfold_LRCV-'+op+'-'+self.dataname+'.csv',index=False,sep=',') 
                    

    def logistic_trainandtest_addvar(self, testsize, n_add):# n_per_group):

        #读入先前已经处理完毕，确认需要导入的字段    
        inclusion = pd.read_csv('..\\UnionPay2include.csv', encoding="gbk")
        var_include = list(inclusion.columns)
        
        #数据标准化        
        data_standardized = pd.DataFrame()        
        for col in self.data.columns: 
            stdev = np.std(self.data[col])
            if stdev != 0:    
                data_standardized[col] = self.data[col] / stdev
        
        data_feature = pd.DataFrame()
        for col in var_include:
            data_feature[col] = data_standardized[col]
        
        data_target = (self.data['风险得分']/10).astype(int)
        #分割数据集为训练集和测试集
        X_train, X_test, Y_train, Y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)               
            
        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train, Y_train)  
        #predicted = classifier.predict(X_test)
        scores = classifier.predict(X_train)
        
        #损失函数
        loss_orig = ((Y_train - scores)**2).sum()
        num_var_orig = X_train.shape[1]

        print('number of variables', num_var_orig)
        print('loss value now', loss_orig)

                       
        # 可能会添加的变量集合
        X_train_pool = list(data_standardized.columns)
        X_train_pool.remove('风险得分')
        for col in X_train.columns:
            X_train_pool.remove(col)

        random.shuffle(X_train_pool)

        # 开始添加变量      
        num_added = 0
        
        for col in X_train_pool:     
            if num_added < n_add:
                X_train_temp = X_train.copy()
                X_train_temp[col] = data_standardized[col]
                classifier.fit(X_train_temp, Y_train)  
                scores_temp = classifier.predict(X_train_temp)
                loss_temp = ((Y_train - scores_temp)**2).sum()

                # 如果能改良损失函数，则将变量引入        
                if loss_temp < loss_orig:
                    loss_orig = loss_temp
                    X_train = X_train_temp.copy()
                    print(col, 'added')
                    print('loss value now', loss_temp)
                    num_added = num_added + 1                    
                    print('number of variables', num_var_orig + num_added)
                    var_include.append(col)
                else:
                    print(col, ': Jesus Christ!')
        
        # 将变量名写入文件
        import csv
        csvfile = open("..\\UnionPay2include_output.csv", "w")
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(var_include)
                
        return X_train


    def logistic_trainandtest_removevar(self, testsize, n_remove):# n_per_group):

        #读入先前已经处理完毕，需要开始剔除的字段    
        inclusion = pd.read_csv('..\\UnionPay2include.csv', encoding="gbk")
        var_include = list(inclusion.columns)
        exclusion = pd.read_csv('..\\UnionPay2exclude.csv', encoding="gbk")
        var_exclude = list(exclusion.columns)
        
        #从inclusion中去除exclusion中含有的字段
        var_include = list(set(var_include).difference(set(var_exclude)))
#        for col in var_include:
#            if col in var_exclude:
#                var_include.remove(col)
        
        #数据标准化        
        data_standardized = pd.DataFrame()        
        for col in var_include: 
            stdev = np.std(self.data[col])
            if stdev != 0:    
                data_standardized[col] = self.data[col] / stdev
        
        data_feature = data_standardized.copy()
        
        data_target = (self.data['风险得分']/10).astype(int)
        #分割数据集为训练集和测试集
        X_train, X_test, Y_train, Y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)               
            
        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train, Y_train)  
        #predicted = classifier.predict(X_test)
        scores = classifier.predict(X_train)
        
        #损失函数
        loss_orig = ((Y_train - scores)**2).sum()
        num_var_orig = X_train.shape[1]
        print('number of variables', num_var_orig)
        print('loss value now', loss_orig)

                       
        # 开始删除变量      

        
        for n in range(n_remove):
            max_improvement = 0
            col2remove = None
            to_break = 0
            column_pool = list(X_train.columns)
            random.shuffle(column_pool)
            for col in column_pool:
                X_train_temp = X_train.copy()
                del X_train_temp[col]
                classifier.fit(X_train_temp, Y_train)  
                scores_temp = classifier.predict(X_train_temp)
                loss_temp = ((Y_train - scores_temp)**2).sum()
                improvement = loss_orig - loss_temp
                print(col,': Jesus Christ!')

                # 删除对损失函数改良最大的变量，直至无法改良      
                if improvement >= max_improvement:
                    max_improvement = improvement
                    col2remove = col
                    print('Black Sheep Found! Congratulations!')
                    
                    # 只要出现可删除变量，就立即删除并进入下次删除评定阶段
                    loss_orig = loss_orig - max_improvement
                    del X_train[col2remove]
                    print(col2remove, 'removed')
                    print('loss value now', loss_orig)
                    print(n+1, ' variables removed ', num_var_orig-n-1, ' remain')
                    to_break = 1
                    break                    
            
            if to_break == 0:
                print('no further improvement could be made')
                break 
            
            
            
            
            '''        
            # 从所有可删除的变量中末位淘汰
            if max_improvement > 0:
                loss_orig = loss_orig - max_improvement
                del X_train[col2remove]
                print(col2remove, 'removed')
                print('loss value now', loss_orig)
                print(n+1, ' variables removed')
            else:
                print('no further improvement could be made')
                break
            '''
        
        # 将变量名写入文件
        import csv
        csvfile = open("..\\UnionPay2include_output.csv", "w")
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(X_train.columns)
        print(pd.datetime.now())
                
        return X_train
    

    def pca_standard(self, testsize, n_eig, n_remove):
        
        X_train_start = self.data.ix[:, self.data.columns != '风险得分']

        X_train = pd.DataFrame()

        for col in X_train_start.columns:
            datatemp = X_train_start[col].copy()

            stdev = np.std(datatemp)
            mean = np.mean(datatemp)
            if stdev != 0:
                X_train[col] = (datatemp - mean) / stdev 
        
        matrix = np.dot(np.transpose(X_train), X_train)/(X_train.shape[0]-1)
        eigenvalues, eigenfunctions = np.linalg.eig(matrix)
        
        eigenvalues = eigenvalues.real
        eigenfunctions = eigenfunctions.real
                
        eigenfunc_trunc = eigenfunctions[:, :n_eig]
        data_feature = np.dot(X_train, eigenfunc_trunc)
        
        print('PCA complete')
        """
        classifier = PCA(n_components = n_eig, copy = True)
           
        data_feature = classifier.fit_transform(X_train)
        
        explained = classifier.explained_variance_ratio_                       
        
        print('PCA complete')
        """
        
                
        
        data_target = (self.data['风险得分']/10).astype(int)
        
        X_train = pd.DataFrame(data_feature)
        Y_train = data_target.copy()

        inclusion = pd.read_csv('..\\UnionPay_pca_coeff_input.csv', encoding="gbk")
        var_include = list(inclusion.columns)


#        print(X_train.columns)
        X_train_new = pd.DataFrame()
        
        for col in var_include:
            col = int(col)
            X_train_new[col] = X_train[col]
            
        X_train = X_train_new.copy()

        #分割数据集为训练集和测试集
#        X_train, X_test, Y_train, Y_test = train_test_split(data_feature, data_target, test_size=testsize, random_state=0)               
        
        #训练并预测模型
        classifier = LogisticRegression()  # 使用类，参数全是默认的
        classifier.fit(X_train, Y_train)  
        #predicted = classifier.predict(X_test)
        scores = classifier.predict(X_train)
        
        #损失函数
        loss_orig = ((Y_train - scores)**2).sum()
        num_var_orig = X_train.shape[1]
        print('number of variables', num_var_orig)
        print('loss value now', loss_orig)        
                
        
        for n in range(n_remove):
            max_improvement = 0
            n_col2remove = None
            to_break = 0
            column_pool = list(X_train.columns)
            random.shuffle(column_pool)
            for n_col in column_pool:
                X_train_temp = X_train.copy()
                del X_train_temp[n_col]
                classifier.fit(X_train_temp, Y_train)  
                scores_temp = classifier.predict(X_train_temp)
                loss_temp = ((Y_train - scores_temp)**2).sum()
                improvement = loss_orig - loss_temp
                print(n_col+1,': Jesus Christ!')

                # 删除对损失函数改良最大的变量，直至无法改良      
                if improvement >= max_improvement:
                    max_improvement = improvement
                    n_col2remove = n_col
                    print('Black Sheep Found! Congratulations!')
                    
                    # 只要出现可删除变量，就立即删除并进入下次删除评定阶段
                    loss_orig = loss_orig - max_improvement
                    del X_train[n_col]

                    print('The ', n_col2remove+1, ' th variable removed (after PCA)')
                    print('loss value now', loss_orig)
                    print(n+1, ' variables removed ', num_var_orig-n-1, ' remain')
                    to_break = 1
                    break                    
            
            if to_break == 0:
                print('no further improvement could be made')
                break 
        
        import csv
        csvfile = open("..\\UnionPay_pca_coeff_output.csv", "w")
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(X_train.columns)
        print(pd.datetime.now())

        return X_train, data_feature#, explained

