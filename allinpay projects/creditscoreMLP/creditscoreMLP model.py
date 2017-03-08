# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from creditscoreMLP.creditscoreMLP import CreditScoreMLP


##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################
dataname = 'HMEQ'
#dataname = 'german'
#dataname = 'taiwancredit'
MLPmodel = CreditScoreMLP(dataname)
self = MLPmodel

##############################################################################
##############################################################################
#二，设置模型参数
##############################################################################
##############################################################################
testsize = 0.25
hidden_layer_sizes = (64,)
#hidden_layer_sizes = (64,32,)
activation = 'relu'
#activation = 'logistic'
alpha = 0.0001
nsplit = 5
cv = 10
varthreshold = 0.2
#粗分类时聚类的数量
nclusters=10
#粗分类时聚类的方法,kmeans,DBSCAN,Birch，quantile(等分位数划分)，None(等距划分)
cmethod = 'kmeans'
#cmethod = None
#cmethod = 'quantile'
#cmethod = 'Birch'

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#1，不筛选变量的完整模型
feature_sel = 'origin'
#单次的train and test
predresult = self.MLP_trainandtest(testsize, cv, feature_sel, varthreshold, activation, alpha, *hidden_layer_sizes, nclusters=nclusters, cmethod=cmethod)
#K重train and test
predresult = self.MLP_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, activation, alpha, *hidden_layer_sizes, nclusters=nclusters, cmethod=cmethod)

#2，VarianceThreshold过滤变量
feature_sel = "VarianceThreshold"
#单次的train and test
predresult = self.MLP_trainandtest(testsize, cv, feature_sel, varthreshold, activation, alpha, *hidden_layer_sizes, nclusters=nclusters, cmethod=cmethod)
#K重train and test
predresult = self.MLP_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, activation, alpha, *hidden_layer_sizes, nclusters=nclusters, cmethod=cmethod)

#3，RFECV递归+CV选择变量
feature_sel = "RFECV"
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




