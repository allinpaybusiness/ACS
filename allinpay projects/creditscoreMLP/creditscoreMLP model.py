# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscoreMLP.classMLP
#reload(creditscoreMLP.classMLP)

##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################
#dataname = 'HMEQ'
#dataname = 'german'
dataname = 'taiwancredit'
MLPmodel = creditscoreMLP.classMLP.CreditScoreMLP(dataname)
self = MLPmodel

##############################################################################
##############################################################################
#二，设置模型参数
##############################################################################
##############################################################################
#1,粗分类和woe转换设置
#粗分类时聚类的数量
nclusters=100
#粗分类时聚类的方法,kmeans,DBSCAN,Birch，quantile(等分位数划分)，None(等距划分)
#cmethod = 'equal'
cmethod = 'quantile'
#cmethod = 'kmeans'
#cmethod = 'Birch'
#method = 'DBSCAN'
#2，训练集和测试集的划分
# 测试样本大小
testsize = 0.25
#交叉检验法分割数量
nsplit = 5
#3，变量筛选设置
feature_sel = 'origin'
#feature_sel = "VarianceThreshold"
#feature_sel = "RFECV"
#feature_sel == "SelectFromModel"
#feature_sel == "SelectKBest"
cv = 10
varthreshold = 0.2
#4，MLP算法设置
hidden_layer_sizes = (64,)
#hidden_layer_sizes = (64,32,)
activation = 'relu'
#activation = 'logistic'
alpha = 0.0001


##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#单次的train and test
predresult = self.MLP_trainandtest(testsize, cv, feature_sel, varthreshold, activation, alpha, *hidden_layer_sizes, nclusters=nclusters, cmethod=cmethod)
#K重train and test
predresult = self.MLP_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, activation, alpha, *hidden_layer_sizes, nclusters=nclusters, cmethod=cmethod)

##############################################################################
##############################################################################
#四，模型预测结果评估
##############################################################################
##############################################################################
self.modelmetrics_scores(predresult)




