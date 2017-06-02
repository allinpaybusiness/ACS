# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd


def fyz_pred(parameters):   
    #生成X_test
    X_test = pd.DataFrame(parameters, index=[0])

    X_test['rsk_score'] = pd.to_numeric(X_test['rsk_score'], errors='coerce')
    
    riskscore = 300 + X_test.ix[0, 'rsk_score'] * 0.7
    
    return riskscore

               