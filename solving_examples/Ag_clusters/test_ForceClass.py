# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:02:45 2024

@author: n.patsalidis
"""
# 1 Import the FF_Develop library
import sys
sys.path.append('../../main_scripts')
import FF_Develop as ff
import pandas as pd
from time import perf_counter
import numpy as np
# 1 end
check_only_analytical = False
verbose =False
num = 10
files = [f'test_Forces{kk}.in' for kk in [3]]
# 2 read the setup file
#files.append('result_even.in')
for file in files:
    setup = ff.Setup_Interfacial_Optimization(file)
    # 2 end 
    
    # 3  Let's read the data
    
    al = ff.al_help()
    
    path_log = 'data'
    path_xyz ='data_xyz'
   
    
    #al.log_to_xyz(path_log, path_xyz)
    
    data = pd.DataFrame()
    #ff.GeneralFunctions.make_dir('distr')
    data = al.data_from_directory(path_xyz)
    al.make_absolute_Energy_to_interaction(data,setup)
    if file == 'result_even.in' :
        path ='Ageven/data'
        data = pd.DataFrame()
        
        for n in range(num+1):
            print('Reading iteration {:d} '.format(n))
            sys.stdout.flush()
            
            path_xyz ='{:s}/D{:d}'.format(path,n)
            df = al.data_from_directory(path_xyz)
            #al.make_absolute_Energy_to_interaction(df, setup)
            data = data.append(df , ignore_index=True)
    #ff.Data_Manager.distribution(data['Energy'],'distr/data{:d}.png'.format(n))
    # 3 end - data are read in the dataframe "data"
    
    # 4 clean the data
    data = al.clean_data(data,setup)
    
    #0ff.Data_Manager(data,setup).distribution('Energy')
    # 4 end
    ff.al_help.make_interactions(data,setup)
    dataMan = ff.Data_Manager(data, setup)
    train_indexes, test_indexes = dataMan.train_development_split()
    
    optimizer = ff.FF_Optimizer(data,train_indexes,test_indexes, setup)
    
    optimizer.test_ForceClass(which='init',epsilon=1e-3,random_tries=10,
                              verbose=verbose,seed=12,
                              check_only_analytical_forces=check_only_analytical) 
