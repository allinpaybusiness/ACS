# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscore_TLSW_fyz.creditscore_randomforest
reload(creditscore_TLSW_fyz.creditscore_randomforest)


##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################

dataname = 'suanhua'
tlswmodel = creditscore_TLSW_fyz.creditscore_randomforest.TLSWscoring_randomforest(dataname)
self = tlswmodel

##############################################################################
##############################################################################
#二，设置模型参数
##############################################################################
##############################################################################
#1,粗分类和woe转换设置
#根据外部分数对好坏客户的分类界限
unionscores = False
cutscore = 600
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
#5，决策树算法设置
ntrees = 200
nodes = 5
rfmethod = 'RandomForest'
#rfmethod = 'ExtraTrees'
#rfmethod = 'GradientBoosting'

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#单次的train and test
predresult = self.RF_trainandtest(unionscores, cutscore, testsize, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod)
#K重train and test
predresult = self.RF_trainandtest_kfold(unionscores, nsplit, cutscore, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod)

#单次的train and test：bagging
predresult = self.RF_bagging_trainandtest(unionscores, cutscore, testsize, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod)
#K重train and test：bagging
predresult = self.RF_bagging_trainandtest_kfold(unionscores, nsplit, cutscore, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod)



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

##############################################################################
##############################################################################
#五，生产：保存模型，提取模型，输出预测结果
##############################################################################
##############################################################################
import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscore_TLSW_fyz.creditscore_randomforest
reload(creditscore_TLSW_fyz.creditscore_randomforest)

dataname = 'suanhua'
label = 'fyz_randomforest'
tlswmodel = creditscore_TLSW_fyz.creditscore_randomforest.TLSWscoring_randomforest(dataname, label)
self = tlswmodel

unionscores = False
cutscore = 600
nclusters=10
cmethod = 'quantile'#暂时只支持quantile 或者equal
testsize = 0
nsplit = 5
feature_sel = 'origin'
cv = 10
varthreshold = 0.2
resmethod = None#暂时不支持样本重采样
ntrees = 200
nodes = 5
rfmethod = 'RandomForest'


predresult = self.RF_trainandtest(unionscores, cutscore, testsize, cv, feature_sel, varthreshold, ntrees, nodes, rfmethod, nclusters, cmethod, resmethod)

self.modelmetrics_scores(predresult)

riskcontrol_cost = 0.01
lend_rate = 0.18
borrow_rate = 0.07
profit_p0 = self.maxprofit_p0(predresult, riskcontrol_cost, lend_rate, borrow_rate)
metrics_p = self.pred_feature_analysis(predresult)





