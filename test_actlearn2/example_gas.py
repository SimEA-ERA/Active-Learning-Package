# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:02:45 2024

@author: n.patsalidis
"""
# 1 Import the FF_Develop library
import sys
sys.path.append('../main_scripts')
import FF_Develop as ff
import pandas as pd
from time import perf_counter

# 1 end
gas='NO2'
# 2 read the setup file
setup = ff.Setup_Interfacial_Optimization(f'{gas}.in')
# 2 end 

# 3  Let's read the data

al = ff.al_help()
path_log = f'new_{gas}'
path_xyz = path_log+'xyz'
al.log_to_xyz(path_log, path_xyz)

#al.log_to_xyz(path_log, path_xyz)

data = pd.DataFrame()
#ff.GeneralFunctions.make_dir('distr')
data = al.data_from_directory(path_xyz)
al.make_absolute_Energy_to_interaction(data,setup)
#ff.Data_Manager.distribution(data['Energy'],'distr/data{:d}.png'.format(n))
# 3 end - data are read in the dataframe "data"

energies = data['Energy'].to_numpy()
forces_dft = []
nbins = 100
for fo in data['Forces'].values:
    for f in fo:
        for x in f:
            forces_dft.append(x)
from matplotlib import pyplot as plt


# 4 clean the data
#data = al.clean_data(data,setup)


# 5 solve the model
t1 = perf_counter()    
data, current_costs, setup, optimizer = al.solve_model(data,setup)
print('solving time = {:.3e} sec '.format(perf_counter()-t1))
# 5 end

forces = []
nbins = 100
for fo in data['Fclass'].values:
    for f in fo:
        for x in f:
            forces.append(x)
            
_ = plt.figure(figsize=(3.3, 3.3), dpi=300)
plt.yscale('log')
plt.hist(energies, bins=nbins, histtype='step', label='ener', density=True, color='blue')
plt.hist(forces_dft, bins=nbins, histtype='step', label='forces_dft', density=True, color='red')
plt.hist(forces, bins=nbins, histtype='step', label='forces_calc', density=True, color='green')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normalized Histograms')
plt.show()

import numpy as np
jmax = -1
diff_max =0
for j,dat in data.iterrows():
    ftrue =dat['Forces']
    fclass = dat['Fclass']
    for ft,fc in zip(ftrue,fclass):
        diff = np.abs(ft-fc).max()
        if diff >diff_max:
            jmax = j
            diff_max = diff
            
