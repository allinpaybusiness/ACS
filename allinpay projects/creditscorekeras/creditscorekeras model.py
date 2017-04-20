# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscorekeras.classkeras
reload(creditscorekeras.classkeras)

##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################
#dataname = 'HMEQ'
#dataname = 'german'
dataname = 'taiwancredit'
kerasmodel = creditscorekeras.classkeras.CreditScoreKeras(dataname)
self = kerasmodel

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
testsize = 0.3
nsplit = 5
#3，变量筛选设置
feature_sel = 'origin'
#feature_sel = "VarianceThreshold"
#feature_sel = "RFECV"
#feature_sel == "SelectFromModel"
#feature_sel == "SelectKBest"
cv = 10
varthreshold = 0.2
#4,样本重采样
#4.0 不重采样
resmethod = None
#4.1 欠采样 undersampling
#resmethod = 'ClusterCentroids'
#resmethod = 'CondensedNearestNeighbour'
#resmethod = 'NearMiss'
#resmethod = 'RandomUnderSampler'
#4.2 过采样 oversampling
#resmethod = 'ADASYN'
#resmethod = 'RandomOverSampler'
#resmethod = 'SMOTE'
#4.3 过采样欠采样结合
#resmethod = 'SMOTEENN'
#resmethod = 'SMOTETomek'
#5，Keras算法设置
batches = 100
nepoch = 1000
#deepmodel = 'dnn1'
deepmodel = 'dnn2'
#6, 是否对变量做pca变换
pca = True
#pca = False

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#1，不筛选变量的完整模型
#单次的train and test
predresult = self.keras_dnn_trainandtest(testsize, cv, feature_sel, varthreshold, pca, nepoch, batches, nclusters, cmethod, resmethod, deepmodel)
#K重train and test
predresult = self.keras_dnn_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, pca, nepoch, batches, nclusters, cmethod, resmethod, deepmodel)

#2，用SVC过滤keras的预测结果
#单次的train and test
predresult = self.keras_SVC_dnn_trainandtest(testsize, cv, feature_sel, varthreshold, pca, nepoch, batches, nclusters, cmethod, resmethod, deepmodel)
#K重train and test
predresult = self.keras_SVC_dnn_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, pca, nepoch, batches, nclusters, cmethod, resmethod, deepmodel)

##############################################################################
##############################################################################
#四，模型预测结果评估
##############################################################################
##############################################################################
#对模型总体预测能力的评价：
self.modelmetrics_scores(predresult)
#计算最优P0阈值
riskcontrol_cost = 0.01
lend_rate = 0.18
borrow_rate = 0.07
self.maxprofit_p0(predresult, riskcontrol_cost, lend_rate, borrow_rate)













