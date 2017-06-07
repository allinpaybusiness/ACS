# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
sys.path.append("TLSW_pred\\fyzpred01")
sys.path.append("TLSW_pred\\fyzpred02")

#指定模型文件，必须与数据库一致
model = 'fyz_pred_01'

#读入参数
parameters = []
for i in range(1, len(sys.argv)): 
    parameters.append(sys.argv[i])

#执行相应的模型
if model == 'fyz_pred_00':    
    from imp import reload
    import fyz_pred_00
    reload(fyz_pred_00)
    rsk_score = fyz_pred_00.fyz_pred(parameters)
    print(rsk_score)
elif model == 'fyz_pred_01':
    from imp import reload
    import fyz_pred_01
    reload(fyz_pred_01)
    rsk_score = fyz_pred_01.fyz_pred(parameters)
    print(rsk_score)
    
        


    
