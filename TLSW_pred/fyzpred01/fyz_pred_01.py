# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import time
import pandas as pd
from sklearn.externals import joblib

def fyz_pred(parameters):   
    #生成X_test
    X_test = pd.DataFrame(parameters)
    X_test = X_test.transpose()
    X_test.columns = ['idCard', 'mobileNum', 'education', 'maritalStatus', 'company', 'isUseTime', 'telStatus', 'MON_6_var1', 'RFM_6_var1', 'RFM_6_var2'\
                      'RFM_6_var3', 'RFM_6_var5', 'MCC_6_var1', 'LOC_6_var11', 'RFM_6_var6', 'RFM_12_var30', \
                      'RFM_12_var40', 'RFM_12_var47', 'cot_score', 'cna_score', 'cst_score', 'cnt_score', 'chv_score', 'dsi_score', 'rsk_score',\
                      'wlp_score', 'crb_score', 'summary_score', 'cnp_score']
    X_test['phone3'] = str(X_test.ix[0, 'mobileNum'])[0:3]
    X_test['age'] = time.localtime().tm_year - int(X_test.ix[0, 'idCard'][6:10])
    X_test['sexId'] = '1'
    if int(X_test.ix[0, 'idCard'][(len(X_test.ix[0, 'idCard'])-2):(len(X_test.ix[0, 'idCard'])-1)]) % 2 == 0:
        X_test['sexId'] = '2'                  
        
        
    classifier = joblib.load("allinpay projects\\creditscore_TLSW_fyz\\pkl\\" + label + '.pkl')
    
    return X_test.ix[0, 'rsk_score']
               