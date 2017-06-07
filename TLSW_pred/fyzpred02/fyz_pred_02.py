# -*- coding: utf-8 -*-
"""
Spyder Editor

生产模型：creditscore_TLSW_fyz.creditscore_randomforest
"""


import sys
import time
import pandas as pd
import numpy as np
from sklearn.externals import joblib

"""
parameters = {'idCard':'530125198606102726', 'mobileNum':'13619662783', 'education':'高中', 'maritalStatus':'22', 
              'company':'三营村委会', 'MON_6_var1':'5', 'RFM_6_var1':'18499.81', 'RFM_6_var2':'19',
              'RFM_6_var3':'9000', 'RFM_6_var5':'973.67', 'MCC_6_var1':'5', 'LOC_6_var11':'云南省', 'RFM_6_var6':'3', 
              'RFM_12_var30':'6000', 'RFM_12_var40':'0', 'RFM_12_var47':'0', 'cot_score':'654', 'cna_score':'4', 
              'cst_score':'3', 'cnt_score':'4', 'chv_score':'3', 'dsi_score':'351', 'rsk_score':'143',
              'wlp_score':'4', 'crb_score':'0.005230373331', 'summary_score':'496', 'cnp_score':'2'}

"""

def fyz_pred(parameters):   
    #生成X_test
    X_test = pd.DataFrame(parameters, index=[0])

    ###建立测试数据dataframe             
    if int(X_test.loc[0, 'idCard'][(len(X_test.loc[0, 'idCard'])-2):(len(X_test.loc[0, 'idCard'])-1)]) % 2 == 0:
        X_test['sexId'] = '2' 
    else:
        X_test['sexId'] = '1'
    X_test['phone3'] = str(X_test.loc[0, 'mobileNum'])[0:3]
    X_test['age'] = time.localtime().tm_year - int(X_test.loc[0, 'idCard'][6:10])                 
    if X_test.loc[0, 'company'] in ['NULL', '不详', '无', '无业', '待业人员' ]:   
        X_test['company'] = 0
    else:
        X_test['company'] = 1

    X_test['MCC_6_var1'] = pd.to_numeric(X_test['MCC_6_var1'], errors='coerce')
    X_test['MON_6_var1'] = pd.to_numeric(X_test['MON_6_var1'], errors='coerce')
    X_test['RFM_6_var1'] = pd.to_numeric(X_test['RFM_6_var1'], errors='coerce')
    X_test['RFM_6_var2'] = pd.to_numeric(X_test['RFM_6_var2'], errors='coerce')
    X_test['RFM_6_var3'] = pd.to_numeric(X_test['RFM_6_var3'], errors='coerce')
    X_test['RFM_6_var5'] = pd.to_numeric(X_test['RFM_6_var5'], errors='coerce')
    X_test['RFM_12_var30'] = pd.to_numeric(X_test['RFM_12_var30'], errors='coerce')
    X_test['RFM_12_var40'] = pd.to_numeric(X_test['RFM_12_var40'], errors='coerce')
    X_test['RFM_12_var47'] = pd.to_numeric(X_test['RFM_12_var47'], errors='coerce')
    
    X_test = X_test.drop(['idCard', 'mobileNum', 'rsk_score'], axis = 1)
    
    ###处理null值
    fillna_value = joblib.load('TLSW_pred\\fyzpred02\\fillna_value_fyz_randomforest.pkl')
    for col in X_test.columns:
        if any(fillna_value.columns == col):
            X_test[col] = X_test[col].fillna(fillna_value.loc[0, col])
    
    ###提取数据预处理模型    
    binandwoe = joblib.load('TLSW_pred\\fyzpred02\\binandwoe_fyz_randomforest.pkl')   
    cols = binandwoe[0]   
    binmodel = binandwoe[1] 
    woemodel = binandwoe[2]   
    
    ###逐列处理数据
    for col in X_test.columns:
        ###bin
        if col in cols:
            ix = cols.index(col)
            breakpoints = binmodel[ix]
            labels = np.arange(len(breakpoints) - 1)
            X_test[col] = pd.cut(X_test[col],bins=breakpoints,right=True,labels=labels,include_lowest=True)
            X_test[col] = X_test[col].astype('object')
        else:
            X_test[col] = X_test[col].astype('object')
            
        ###woe
        if any(woemodel['col'] == col):
            woecol = woemodel[woemodel['col'] == col]
            if any(woecol['cat'] == X_test.loc[0, col]):
                X_test[col] = woecol.loc[woecol['cat'] == X_test.loc[0, col], 'woe']
            else:
                X_test[col] = 0
        else:
            X_test[col] = 0
        
    ###提取最终参与建模的列
    testcolumns = joblib.load('TLSW_pred\\fyzpred02\\testcolumns_fyz_randomforest.pkl')
    X_test = X_test[testcolumns]
        
    ###提取违约概率模型        
    classifier = joblib.load('TLSW_pred\\fyzpred02\\classifier_fyz_randomforest.pkl')    
    probability = classifier.predict_proba(X_test)
    
    ###转换概率至评分
    riskscore = str(int(300 + probability[0][0] * 700))
    
    return riskscore
               