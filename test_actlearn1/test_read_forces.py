import numpy as np
import sys
sys.path.append('../main_scripts')
import FF_Develop as ff

dm = ff.Data_Manager.read_Gaussian_output('data/L0/Ag4CO2.log',read_forces=True)

print(dm['Forces'][0])
