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
L = 10
dx =0.001
if True:
    params = np.array([ L,0,1,3,-5,20,-20,-15,-5,0],dtype=np.float64)
    m1 = ff.TestPotentials('Bezier',params,0,L,dx,plot=False)
    m1.derivative_check(plot=True,verbose=True,)
    m1.gradient_check(tol=1e0,plot=True,verbose=True,epsilon=1e-4)

if True:
    L /= 10
    params = np.array([L,0,3,3,-5,20,-20,-15,-5,0], dtype=np.float64 )
    m2 = ff.TestPotentials('Bezier',params,0,L,dx)
    m2.gradient_check(tol=1,verbose=True)
    m2.derivative_check(verbose=True)
    
    L*=10
    params = np.array([L,0,31,3,-15,-20,-20,-15,-25,10], dtype=np.float64)
    m3 = ff.TestPotentials('Bezier',params,0,L,dx)
    m3.gradient_check(tol=1e0)
    m3.derivative_check(tol=1e0,verbose=True)

if True:
    params = np.array([ L,0,1,3,-5,20,-20,-15,-5,0],dtype=np.float64)
    m4 = ff.TestPotentials('Bezier',params,0,L,dx)
    times = m4.time_cost(10)
if True:
    m5 = ff.TestPotentials('Bezier',params,0,L,dx)  
    m5.vectorization_scalability(Nt=10,verbose=True,plot=True)
if True:
    params = np.array([ 3.37786357, -9.53988951, 15.56390922,  5.5807232,  -6.44692876,  7.26188354
 -4.97285838,  3.18828564, 17.79433481, 16.5597632,   0.] )
    m3 = ff.TestPotentials('Bezier',params,0,10,dx)
    m3.gradient_check(tol=1e0,verbose=True)
    m3.derivative_check(tol=1e0,verbose=True)