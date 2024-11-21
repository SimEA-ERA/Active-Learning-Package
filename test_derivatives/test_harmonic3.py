# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:06:04 2024

@author: n.patsalidis
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../main_scripts')
import FF_Develop as ff
from time import perf_counter
params = np.array([2.5,10,5,50])
m1 = ff.TestPotentials('harmonic3',params,0,10,0.0001,ignore_high_u=100,plot=True)
m1.gradient_check(tol=1e0,plot=True,verbose=True)
m1.derivative_check(plot=True,verbose=True)
if True:
    params = np.array([2.2,300,-19,5])
    m2 = ff.TestPotentials('harmonic3',params,0,5,0.0001,ignore_high_u=100)
    m2.gradient_check(tol=1)
    m2.derivative_check(verbose=True)
    
    
    params = np.array([2.2,500,-200,128])
    m3 = ff.TestPotentials('harmonic3',params,0,5,0.0001,ignore_high_u=100)
    m3.gradient_check(tol=1e1,verbose=True)
    m3.derivative_check(tol=1e1,verbose=True)
    m3.time_cost()
    m3.vectorization_scalability()