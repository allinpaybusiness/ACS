# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from creditscorekeras.creditscorekeras import CreditScoreKeras


##################
#一，初始化模型数据
#################
dataname = 'HMEQ'
#dataname = 'german'
kerasmodel = CreditScoreKeras(dataname)
self = kerasmodel

#################
#二，设置模型参数
#################
binn = 10
testsize = 0.25
nsplit = 5
cv = 10
bq = True
nepoch = 10
batches = 5

#################
#三，建模并预测
#################
#1，不筛选变量的完整模型
#单次的train and test
predresult = self.keras_dnn_trainandtest(binn, testsize, cv, bq=bq, nepoch=nepoch, batches=batches)

#K重train and test
predresult = self.keras_dnn_trainandtest_kfold(binn, nsplit, cv, bq=bq, nepoch=nepoch, batches=batches)

#2，VarianceThreshold过滤变量
feature_sel = "VarianceThreshold"
varthreshold = 0.2
#单次的train and test
predresult = self.keras_dnn_trainandtest(binn, testsize, cv, feature_sel, varthreshold, bq=bq, nepoch=nepoch, batches=batches)

#K重train and test
predresult = self.keras_dnn_trainandtest_kfold(binn, nsplit, cv, feature_sel, varthreshold, bq=bq, nepoch=nepoch, batches=batches)

#3，RFECV递归+CV选择变量
feature_sel = "RFECV"
#单次的train and test
predresult = self.keras_dnn_trainandtest(binn, testsize, cv, feature_sel, bq=bq, nepoch=nepoch, batches=batches)

#K重train and test
predresult = self.keras_dnn_trainandtest_kfold(binn, nsplit, cv, feature_sel, bq=bq, nepoch=nepoch, batches=batches)

#################
#四，模型预测结果评估
#################
self.modelmetrics_scores(predresult)



