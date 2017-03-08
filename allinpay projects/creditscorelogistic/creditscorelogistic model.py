# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from creditscorelogistic.creditscorelogistic import CreditScoreLogistic


##################
#一，初始化模型数据
#################

#dataname = 'HMEQ'
dataname = 'german'
#dataname = 'taiwancredit'

logisticmodel = CreditScoreLogistic(dataname)
self = logisticmodel

#################
#二，设置模型参数
#################
# WOE 分割数等份
#binn = 10
# 测试样本大小
testsize = 0.25
#交叉检验法分割数量
nsplit = 5
# 变量删选分割数量
cv = 10

#逻辑回归优化方法：liblinear，lbfgs，newton-cg，sag，样本超过10W建议用sag
op = 'liblinear'
#粗分类时聚类的数量
nclusters=10
#粗分类时聚类的方法,kmeans,DBSCAN,Birch，quantile
cmethod = 'Birch'


#################
#三，建模并预测
#################
#####1，不筛选变量的完整模型
feature_sel = "origin"
#1)简单粗分类
#单次的train and test
predresult = self.logistic_trainandtest(testsize, cv, nclusters=nclusters,cmethod=cmethod)
#K重train and test
predresult = self.logistic_trainandtest_kfold(nsplit, cv, nclusters=nclusters,cmethod=cmethod)

#2)聚类粗分类
#单次的train and test
#predresult = self.logistic_trainandtest(testsize, cv,nclusters=nclusters,cmethod=cmethod)
#K重train and test
#predresult = self.logistic_trainandtest_kfold( nsplit, cv, nclusters=nclusters, cmethod=cmethod)

######2，VarianceThreshold过滤变量
feature_sel = "VarianceThreshold"
varthreshold = 0.2
#单次的train and test
predresult = self.logistic_trainandtest(testsize, cv, feature_sel, varthreshold,nclusters=nclusters,cmethod=cmethod)

#K重train and test
predresult = self.logistic_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold,nclusters=nclusters,cmethod=cmethod)

#####3，RFECV递归+CV选择变量
feature_sel = "RFECV"

##### SelectFromModel选择变量
feature_sel = "SelectFromModel"
#单次的train and test

# 遍历测试binn,binn从3到100，本方法已包括模型评估，并且保存到文件中
predresult = self.looplogistic_trainandtest(testsize, cv, feature_sel, cmethod=cmethod)
    
# 传入binn
predresult = self.logistic_trainandtest(testsize, cv, feature_sel, nclusters=nclusters,cmethod=cmethod)

#K重train and test

# 遍历测试binn,binn从3到100，本方法已包括模型评估，并且保存到文件中
predresult = self.looplogistic_trainandtest_kfold(nsplit, cv, feature_sel,cmethod=cmethod )

predresult = self.looplogistic_trainandtest_kfold_LRCV(nsplit, cv, feature_sel,op=op,cmethod=cmethod)

# 遍历测试ncluster,ncluster从3到100，本方法已包括模型评估，并且保存到文件中
predresult = self.looplogistic_trainandtest(testsize, cv, feature_sel,cmethod=cmethod)

predresult = self.looplogistic_trainandtest_kfold(nsplit, cv, feature_sel,cmethod=cmethod)
    
predresult = self.logistic_trainandtest_kfold(nsplit, cv, feature_sel, nclusters=nclusters,cmethod=cmethod)

# 如果不做WOE，生成模型
#predresult = self.logistic_trainandtest_kfold_nowoe(nsplit, cv, feature_sel )

#################
#四，模型预测结果评估
#################
self.modelmetrics_scores(predresult)




