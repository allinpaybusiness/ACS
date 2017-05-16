# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscoreRandomForest.classRandomForest
#reload(creditscoreRandomForest.classRandomForest)


##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################
#dataname = 'HMEQ'
dataname = 'german'
#dataname = 'taiwancredit'
#dataname = 'gmsc'
RFmodel = creditscoreRandomForest.classRandomForest.CreditScoreRandomForest(dataname)
self = RFmodel

##############################################################################
##############################################################################
#二，设置模型参数
##############################################################################
##############################################################################
#1,粗分类和woe转换设置
#粗分类时聚类的数量
nclusters=10
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
#4，决策树算法设置
ntrees = 200
nodes = 1
rfmethod = 'RandomForest'
#rfmethod = 'ExtraTrees'
#rfmethod = 'GradientBoosting'

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#单次的train and test
predresult = self.RF_trainandtest(testsize, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod)
#K重train and test
predresult = self.RF_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod)

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
profit_p0 = self.maxprofit_p0(predresult, riskcontrol_cost, lend_rate, borrow_rate)






