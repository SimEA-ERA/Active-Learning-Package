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

# 1 end

# 2 read the setup file
setup = ff.Setup_Interfacial_Optimization('example1.in')
# 2 end 

# 3  Let's read the data

al = ff.al_help()

path ='Ageven/data'

#al.log_to_xyz(path_log, path_xyz)

data = pd.DataFrame()
num=50
for n in range(num+1):
    print('Reading iteration {:d} '.format(n))
    sys.stdout.flush()
    
    path_xyz ='{:s}/D{:d}'.format(path,n)
    df = al.data_from_directory(path_xyz)
    #al.make_absolute_Energy_to_interaction(df, setup)
    data = data.append(df , ignore_index=True)
    #ff.Data_Manager.distribution(data['Energy'],'distr/data{:d}.png'.format(n))

#ff.GeneralFunctions.make_dir('distr')

al.make_absolute_Energy_to_interaction(data,setup)
#ff.Data_Manager.distribution(data['Energy'],'distr/data{:d}.png'.format(n))
# 3 end - data are read in the dataframe "data"

# 4 clean the data
data = al.clean_data(data,setup)

#0ff.Data_Manager(data,setup).distribution('Energy')
# 4 end

# 5 solve the model
t1 = perf_counter()    
data, errors, optimizer = al.solve_model(data,setup)
print('solving time = {:.3e} '.format(perf_counter()-t1))
# 5 end