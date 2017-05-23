# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import pandas as pd
from sklearn.externals import joblib
    
def fyz_pred(parameters):   
    #生成X_test
    X_test = pd.DataFrame(parameters)
    X_test = X_test.transpose()
    X_test.columns = ['isUseTime', 'phone3', 'maritalStatus', 'education', 'company', 'age', 'cardNum', 'cst_score',\
                   'cnp_score', 'cna_score', 'cnt_score', 'chv_score', 'dsi_score', 'rsk_score', 'MCC_6_var1', 'MON_6_var1', 'RFM_6_var1', 'RFM_6_var2',\
                   'RFM_6_var3', 'RFM_6_var5', 'LOC_6_var11', 'RFM_12_var30', 'RFM_12_var40', 'RFM_12_var47']
    return X_test.ix[0, 'rsk_score']
               