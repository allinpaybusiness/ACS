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
dataname = 'HMEQ'#或'german'
logisticmodel = CreditScoreLogistic(dataname)
self = logisticmodel

#################
#二，设置模型参数
#################
binn = 10
testsize = 0.25
cv = 10

#################
#三，设置模型参数
#################
#1，不筛选变量的完整模型
predresult = self.logistic_trainandtest(binn, testsize, cv)

#2，VarianceThreshold过滤变量
feature_sel = "VarianceThreshold"
varthreshold = 0.2
predtable = self.logistic_trainandtest(binn, testsize, cv, feature_sel, varthreshold)

#3，RFECV递归+CV选择变量
feature_sel = "RFECV"
predtable = self.logistic_trainandtest(binn, testsize, cv, feature_sel)

#################
#四，模型预测结果评估
#################
self.modelmetrics_binary(predtable)




