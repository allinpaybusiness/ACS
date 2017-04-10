# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscorelogistic.classlogistic
reload(creditscorelogistic.classlogistic)

##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################

#dataname = 'HMEQ'
dataname = 'german'
#dataname = 'taiwancredit'
logisticmodel = creditscorelogistic.classlogistic.CreditScoreLogistic(dataname)
self = logisticmodel

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
#5，Logistic算法设置
#逻辑回归优化方法：liblinear，lbfgs，newton-cg，sag，样本超过10W建议用sag
op = 'liblinear'

#6,数据预处理
#preprocess = None
#标准化、将数据缩放至给定的最小值与最大值之间
preprocess = 'MinMaxScaler'
#标准化、将最大的绝对值缩放至单位大小
#preprocess = 'MaxAbsScaler'
#保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据
#preprocess = 'StandardScaler'
#直接将给定数据进行标准化
#preprocess = 'scale'
#归一化、对训练集和测试集的拟合和转换,#向量空间模型(VSM)的基础，用于文本分类和聚类处理。
#preprocess = 'Normalizer'
#归一化、对指定数据进行转换
#preprocess = 'normalize'
#二值化、将数值型的特征值转换为布尔值，可以用于概率估计
#preprocess = 'Binarizer'

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#单次的train and test
predresult = self.logistic_trainandtest(testsize, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, preprocess)
#K重train and test
predresult = self.logistic_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, preprocess)

# 遍历测试binn,binn从3到100，本方法已包括模型评估，并且保存到文件中
predresult = self.looplogistic_trainandtest(testsize, cv, feature_sel, cmethod=cmethod)
# 遍历测试binn,binn从3到100，本方法已包括模型评估，并且保存到文件中
predresult = self.looplogistic_trainandtest_kfold(nsplit, cv, feature_sel,cmethod=cmethod )
predresult = self.looplogistic_trainandtest_kfold_LRCV(nsplit, cv, feature_sel,op=op,cmethod=cmethod)

# 遍历测试ncluster,ncluster从3到100，本方法已包括模型评估，并且保存到文件中
predresult = self.looplogistic_trainandtest(testsize, cv, feature_sel,cmethod=cmethod)
predresult = self.looplogistic_trainandtest_kfold(nsplit, cv, feature_sel,cmethod=cmethod)
    

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


