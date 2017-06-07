# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import classlogistic
reload(classlogistic)

##############################################################################
##############################################################################
#一，初始化模型数据
##############################################################################
##############################################################################

#dataname = 'HMEQ'
#dataname = 'german'
#dataname = 'taiwancredit'
#dataname = 'UnionPay'
dataname = 'gmsc'
logisticmodel = classlogistic.CreditScoreLogistic(dataname)
self = logisticmodel

##############################################################################
##############################################################################
#二，设置模型参数
##############################################################################
##############################################################################
#1,粗分类和woe转换设置
#粗分类时聚类的数量
#nclusters=10
#粗分类时聚类的方法,kmeans,DBSCAN,Birch，quantile(等分位数划分)，None(等距划分)
#cmethod = 'equal'
#cmethod = 'quantile'
#cmethod = 'kmeans'
#cmethod = 'Birch'
#method = 'DBSCAN'
#2，训练集和测试集的划分
# 测试样本大小
#testsize = 1
#交叉检验法分割数量
#nsplit = 5
#3，变量筛选设置
#feature_sel = 'origin'
#feature_sel = "VarianceThreshold"
#feature_sel = "RFECV"
#feature_sel == "SelectFromModel"
#feature_sel == "SelectKBest"
#cv = 10
#varthreshold = 0.2
# 希望由模型自动剔除的（相对无关的）特征变量数目
#nvar2exclude = 10
#4,样本重采样
#4.0 不重采样
#resmethod = None
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
#op = 'liblinear'
#6 pca变量保留删除设置
#n_add = 10 #欲添加的特征变量数
#n_remove = 10 #欲删除的特征变量数
#n_eig = 100 #欲保留的主成分数
#n_per_group = 50

##############################################################################
##############################################################################
#三，建模并预测
##############################################################################
##############################################################################
#单次的train and test
#predresult = self.logistic_trainandtest(testsize, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod)
#K重train and test
#predresult = self.logistic_trainandtest_kfold(nsplit, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod)

# 遍历测试binn,binn从3到100，本方法已包括模型评估，并且保存到文件中
#predresult = self.looplogistic_trainandtest(testsize, cv, feature_sel, cmethod=cmethod)
# 遍历测试binn,binn从3到100，本方法已包括模型评估，并且保存到文件中
#predresult = self.looplogistic_trainandtest_kfold(nsplit, cv, feature_sel,cmethod=cmethod )
#predresult = self.looplogistic_trainandtest_kfold_LRCV(nsplit, cv, feature_sel,op=op,cmethod=cmethod)

# 遍历测试ncluster,ncluster从3到100，本方法已包括模型评估，并且保存到文件中
#predresult = self.looplogistic_trainandtest(testsize, cv, feature_sel,cmethod=cmethod)
#predresult = self.looplogistic_trainandtest_kfold(nsplit, cv, feature_sel,cmethod=cmethod)

#var_in  = self.logistic_trainandtest_addvar(testsize, n_add)
#var_in  = self.logistic_trainandtest_removevar(testsize, n_remove)

#newp, origp = self.pca_standard(testsize, n_eig, n_remove)
#orig, new, eigval, eigfunc = self.pca(testsize, n_eig, n_remove)

#, cv, feature_sel, varthreshold, nclusters, cmethod, resmethod, nvar2exclude)
    

##############################################################################
##############################################################################
#四，模型预测结果评估
##############################################################################
##############################################################################
#对模型总体预测能力的评价：
"""
#计算最优P0阈值
#riskcontrol_cost = 0.01
#lend_rate = 0.18
#borrow_rate = 0.07
#self.maxprofit_p0(predresult, riskcontrol_cost, lend_rate, borrow_rate)


"""