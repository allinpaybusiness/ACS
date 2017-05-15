# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:56:45 2017

@author: yesf
"""
##############################################################################
#为方便与logistic模型结果的比对，评分卡为“颠倒型”，也就是分数越高，违约可能越大
#散度最大化的基本想法是将评分卡设置为特征的线性组合后，最大化好人群和坏人群的散度
#好人群和坏人群的散度定义为：
# 目标函数：max [f(s,G)-f(s,B)]ln[f(s,G)/f(s,B)]ds (Divergence)
# 其简化形式为正态分布化后的散度(Divergence_Normal)，或马氏距离(Mahal_Dist)
# 评分卡形式依然为线性组合：sum(C_j*X_ij)
##############################################################################

import sys;
sys.path.append("allinpay projects/creditscoredivmax")
from imp import reload
import classdivmax
reload(classdivmax)

##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################

#dataname = 'HMEQ'
dataname = 'german'
#dataname = 'taiwancredit'
DivMaxmodel = classdivmax.CreditScoreDivergenceMax(dataname)
self = DivMaxmodel

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
# 测试样本大小
testsize = 0.5
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
#5 Divergence Maximization 的变量

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#单次的train and test
predresult = self.DivMax_trainandtest(testsize, cv, feature_sel, varthreshold,\
            nclusters, cmethod, resmethod)
#K重train and test
#predresult = self.LinProg_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod)
#check = ((predresult.target!=predresult.predicted)*abs(predresult.probability-cutoff_bad)).sum()
#print('check=', check)
#ratio = (predresult.target==predresult.predicted).sum()/predresult.shape[0]
#print('correct ratio=', ratio)
##############################################################################
##############################################################################
#四，模型预测结果评估
##############################################################################
##############################################################################
#对模型总体预测能力的评价：
#self.modelmetrics_binary(predresult)
self.modelmetrics_scores(predresult)