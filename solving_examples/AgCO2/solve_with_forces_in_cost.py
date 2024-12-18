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

# 2 read the setup file
setup = ff.Setup_Interfacial_Optimization('runned_test1.in')
# 2 end 

# 3  Let's read the data

al = ff.al_help()

path_xyz ='test_data2'

#al.log_to_xyz(path_log, path_xyz)

data = pd.DataFrame()
#ff.GeneralFunctions.make_dir('distr')
data = al.data_from_directory(path_xyz)
al.make_absolute_Energy_to_interaction(data,setup)
#ff.Data_Manager.distribution(data['Energy'],'distr/data{:d}.png'.format(n))
# 3 end - data are read in the dataframe "data"

# 4 clean the data
data = al.clean_data(data,setup)
al.make_interactions(data, setup)

dataMan = ff.Data_Manager(data,setup)


train_indexes, dev_indexes = dataMan.train_development_split()
       
optimizer = ff.FF_Optimizer(data,train_indexes, dev_indexes,setup)

sudoForces = optimizer.test_ForceClass('init',check_only_analytical_forces=True,verbose=False) 
natoms_per_point = data['natoms'].to_numpy()

data['Forces'] = sudoForces.values()
#adding noise
#for j,dat in data.iterrows():
#   dat['Forces'] += np.random.normal(0,5,size=dat['Forces'].shape)

# 5 solve the model
t1 = perf_counter()    
data, errors, optimizer = al.solve_model(data,setup)
print('solving time = {:.3e} sec '.format(perf_counter()-t1))

# 5 end