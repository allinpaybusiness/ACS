# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from creditscoreMLP.creditscoreMLP import CreditScoreMLP


##################
#一，初始化模型数据
#################
dataname = 'HMEQ'
#dataname = 'german'
#dataname = 'taiwancredit'
MLPmodel = CreditScoreMLP(dataname)
self = MLPmodel

#################
#二，设置模型参数
#################
binn = 10
testsize = 0.25
hidden_layer_sizes = (64,)
nsplit = 5
cv = 10
varthreshold = 0.2
bq = True

#################
#三，建模并预测
#################
#1，不筛选变量的完整模型
feature_sel = 'origin'
#单次的train and test
predresult = self.MLP_trainandtest(binn, testsize, cv, feature_sel, varthreshold, bq, *hidden_layer_sizes)

#K重train and test
predresult = self.MLP_trainandtest_kfold(binn, nsplit, cv, feature_sel, varthreshold, bq, *hidden_layer_sizes)

#2，VarianceThreshold过滤变量
feature_sel = "VarianceThreshold"
#单次的train and test
predresult = self.MLP_trainandtest(binn, testsize, cv, feature_sel, varthreshold, bq, *hidden_layer_sizes)

#K重train and test
predresult = self.MLP_trainandtest_kfold(binn, nsplit, cv, feature_sel, varthreshold, bq, *hidden_layer_sizes)

#3，RFECV递归+CV选择变量
feature_sel = "RFECV"
#单次的train and test
predresult = self.MLP_trainandtest(binn, testsize, cv, feature_sel, varthreshold, bq, *hidden_layer_sizes)

#K重train and test
predresult = self.MLP_trainandtest_kfold(binn, nsplit, cv, feature_sel, varthreshold, bq, *hidden_layer_sizes)

#################
#四，模型预测结果评估
#################
self.modelmetrics_scores(predresult)




