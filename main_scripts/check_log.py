import numpy as np
import sys
import FF_Develop as ff
import os
path = 'gases'
num = 3
path_log = '{:s}/L{:d}'.format(path,num)

os.system('bash extract_logfiles.sh {:d} {:s}'.format(num,path))

al = ff.al_help

path_xyz ='{:s}/CHECK{:d}'.format(path,num)

os.system(f'mkdir -p {path_xyz}')

al.log_to_xyz(path_log, path_xyz, read_forces=False)

