# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:01:33 2017

@author: Administrator
"""

import numpy as np
import pandas as pd

def Mahal_Dist(list1, list2):
# This function calculates the mahal distance of good points (in list1) and bad points (in list2)   
# by assuming the good and bad points follow the gaussian distribution and same stdev  

        mean1 = np.mean(list1)
        mean2 = np.mean(list2)
        stdev1 = np.std(list1)
        stdev2 = np.std(list2)
        stdev = (stdev1 + stdev2)/2

        dist = abs(mean1 - mean2) / stdev
      
        return dist

def Divergence_normal(list1, list2):
# This function calculates the divergence of good points (in list1) and bad points (in list2)   
# by assuming the good and bad points follow the gaussian distribution    

        mean1 = np.mean(list1)
        mean2 = np.mean(list2)
        stdev1 = np.std(list1)
        stdev2 = np.std(list2)

        dist = ((stdev1**2 + stdev2**2)*(mean1 - mean2)**2 + (stdev1**2 - stdev2**2)**2) / (2 * stdev1**2 * stdev2**2)
      
        return dist


def Divergence(list1, list2):
# This function calculates the divergence of good points (in list1) and bad points (in list2)   
    
        max_list = max(max(list1),max(list2))
        min_list = min(min(list1),min(list2))
        sample_size = len(list1) + len(list2)
        num_cat = max(int(sample_size / 20),1)

        breakpoints = np.arange(min_list, max_list, (max_list-min_list)/num_cat) 
        breakpoints = np.append(breakpoints, max_list)
        labels = np.arange(len(breakpoints) - 1)
        datacut1 = pd.cut(list1,bins=breakpoints,right=True,labels=labels,include_lowest=True)
        datacut2 = pd.cut(list2,bins=breakpoints,right=True,labels=labels,include_lowest=True)
        
        div = 0
        for cat in labels:
            dist1 = (datacut1 == cat).sum() / len(list1)
            dist2 = (datacut2 == cat).sum() / len(list2)
            div = div + (dist1 - dist2) * np.log (dist1/dist2)
        
        
        
        
        return div
    
    
    