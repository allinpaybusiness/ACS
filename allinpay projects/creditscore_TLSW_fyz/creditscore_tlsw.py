# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys;
sys.path.append("allinpay projects")
from imp import reload
import creditscore.creditscore
reload(creditscore.creditscore)
from creditscore.creditscore import CreditScore
import pandas as pd
import time




class TLSWscoring(CreditScore):
    
    def __init__(self, dataname):
        super(TLSWscoring, self).__init__(dataname)
        self.dataname =  dataname
        self.data = self.suanhuadata()

    def suanhuadata_raw(self):
        temp2001 = self.data_from_mysql('T_POSITION_2001')
        temp2001 = temp2001.drop(['status', 'successStatus', 'errors', 'insertTime', 'source'], axis = 1)
        temp3012 = self.data_from_mysql('T_MOBILE_CREDIT_3012')
        temp3012 = temp3012.drop(['status', 'successStatus', 'errors', 'insertTime', 'source'], axis = 1)
        temp5001 = self.data_from_mysql('T_CONSUMPTION_ACT_5001')
        temp5001 = temp5001.drop(['status', 'success', 'errors', 'insertTime', 'source', 'cot_score', 'cna_score'], axis = 1)
        temp5003 = self.data_from_mysql('T_CONSUMPTION_INFO_5003')
        temp5003 = temp5003.drop(['status', 'successStatus', 'errors', 'insertTime', 'source'], axis = 1)
        data = pd.merge(temp2001, temp3012)
        data = pd.merge(data, temp5001)
        data = pd.merge(data, temp5003)
        
        return data
        
    def suanhuadata(self):
        temp2001 = self.data_from_mysql('T_POSITION_2001')
        temp2001 = temp2001.drop_duplicates(['idCard'])
        data = temp2001[['name', 'idCard', 'mobileNum', 'sexId', 'maritalStatus', 'education', 'company']].copy()
        data['sexId'] = '1' 
        data['maritalStatus'] = data['maritalStatus'].astype('object')
        data['education'] = data['education'].astype('object')
        data['age'] = 0
        data['company'] = data['company'].fillna(0)
        data['company'] = data['company'].replace({'不详':0})        
        data['company'] = data['company'].replace({'无':0})
        data.ix[data['company'] != 0, 'company'] = 1
        
        for i in data.index:
            data.ix[i, 'phone3'] = str(data.ix[i, 'mobileNum'])[0:3]
            data.ix[i, 'age'] = time.localtime().tm_year - int(data.ix[i, 'idCard'][6:10])
            if int(data.ix[i, 'idCard'][(len(data.ix[i, 'idCard'])-2):(len(data.ix[i, 'idCard'])-1)]) % 2 == 0:
                data.ix[i, 'sexId'] = '2'
        '''    
        temp3012 = self.data_from_mysql('T_MOBILE_CREDIT_3012')
        temp = temp3012[['idCard', 'mobileNum', 'isUseTime']].copy()
        temp['phone3'] = temp['mobileNum']
        for i in temp.index:
            temp.ix[i, 'phone3'] = str(temp.ix[i, 'phone3'])[0:3]        
        data = pd.merge(temp, data)
        '''
        temp5003 = self.data_from_mysql('T_CONSUMPTION_INFO_5003')
        temp5003 = temp5003.drop_duplicates(['idCard'])
        temp = temp5003[['idCard', 'cardNum', 'MCC_6_var1', 'MON_6_var1', 'RFM_6_var1', 'RFM_6_var2', 'RFM_6_var3', 'RFM_6_var5', 'LOC_6_var11', \
                        'RFM_12_var30', 'RFM_12_var40', 'RFM_12_var47']].copy()
        temp['MCC_6_var1'] = pd.to_numeric(temp['MCC_6_var1'], errors='coerce')
        temp['MON_6_var1'] = pd.to_numeric(temp['MON_6_var1'], errors='coerce')
        temp['RFM_6_var1'] = pd.to_numeric(temp['RFM_6_var1'], errors='coerce')
        temp['RFM_6_var2'] = pd.to_numeric(temp['RFM_6_var2'], errors='coerce')
        temp['RFM_6_var3'] = pd.to_numeric(temp['RFM_6_var3'], errors='coerce')
        temp['RFM_6_var5'] = pd.to_numeric(temp['RFM_6_var5'], errors='coerce')
        temp['RFM_12_var30'] = pd.to_numeric(temp['RFM_12_var30'], errors='coerce')
        temp['RFM_12_var40'] = pd.to_numeric(temp['RFM_12_var40'], errors='coerce')
        temp['RFM_12_var47'] = pd.to_numeric(temp['RFM_12_var47'], errors='coerce')
        data = pd.merge(data, temp)

        temp5001 = self.data_from_mysql('T_CONSUMPTION_ACT_5001')
        temp5001 = temp5001.drop_duplicates(['idCard'])
        temp = temp5001[['idCard', 'cardNum', 'cst_score', 'cnp_score', 'cnt_score', 'chv_score', 'dsi_score', 'rsk_score']].copy()
        temp['dsi_score'] = pd.to_numeric(temp['dsi_score'], errors='coerce')
        temp['rsk_score'] = pd.to_numeric(temp['rsk_score'], errors='coerce')
        data = pd.merge(data, temp)
        
        for col in data.columns:
            if data[col].dtype == 'O':
                data[col] = data[col].fillna('NA')
            else:
                data[col] = data[col].fillna(data[col].mean())
        
        return data
        
    def data_analysis(self):
        data = self.data
        temp = data[data['rsk_score'] < 9990]
        grouped = temp['rsk_score'].groupby(temp['education'])
        tempmean = pd.DataFrame(grouped.mean())
        tempmean.columns = ['rsk_mean']
        tempcount = pd.DataFrame(grouped.count())
        tempcount.columns = ['rsk_count']
        stat_education = pd.merge(tempmean, tempcount,left_index=True,right_index=True)
        stat_education = stat_education.sort_values(by = ['rsk_mean'], ascending = [0])
        
        temp = data[data['rsk_score'] < 9990]
        grouped = temp['rsk_score'].groupby(temp['sexId'])
        tempmean = pd.DataFrame(grouped.mean())
        tempmean.columns = ['rsk_mean']
        tempcount = pd.DataFrame(grouped.count())
        tempcount.columns = ['rsk_count']
        stat_sex = pd.merge(tempmean, tempcount,left_index=True,right_index=True)
        stat_sex = stat_sex.sort_values(by = ['rsk_mean'], ascending = [0]) 
        
        temp = data[data['rsk_score'] < 9990]
        grouped = temp['rsk_score'].groupby(temp['company'])
        tempmean = pd.DataFrame(grouped.mean())
        tempmean.columns = ['rsk_mean']
        tempcount = pd.DataFrame(grouped.count())
        tempcount.columns = ['rsk_count']
        stat_company = pd.merge(tempmean, tempcount,left_index=True,right_index=True)
        stat_company = stat_company.sort_values(by = ['rsk_mean'], ascending = [0])  
        
        temp = data[data['rsk_score'] < 9990]
        grouped = temp['rsk_score'].groupby(temp['maritalStatus'])
        tempmean = pd.DataFrame(grouped.mean())
        tempmean.columns = ['rsk_mean']
        tempcount = pd.DataFrame(grouped.count())
        tempcount.columns = ['rsk_count']
        stat_marital = pd.merge(tempmean, tempcount,left_index=True,right_index=True)
        stat_marital = stat_marital.sort_values(by = ['rsk_mean'], ascending = [0])
        
        #data = self.data.copy()

         
        

        
    
        
        
        
        
       
        
        
        
