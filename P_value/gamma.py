# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:54:49 2018

@author: zxpay
"""
import numpy as np

def GammaFunc(x):
    data = 0
    for k in range(100):
        data += ((k**(x-1))/np.exp(k))
    return data

