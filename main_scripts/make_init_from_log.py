import numpy as np
import sys
import FF_Develop as ff
import os
path = 'gases'
num = 3
path_log = '{:s}'.format(path)

al = ff.al_help
path_xyz ='{:s}/D{:d}'.format(path,num)

os.system(f'mkdir -p {path_xyz}')

al.log_to_xyz(path_log, path_xyz, read_forces=False)

data = al.data_from_directory(path_xyz)

al.write_drun('{:s}/J{:d}'.format(path,num), data,len(data),num)

c1 = f'bash make_rundirs.sh {num} {path}'
c2 = f'cd {path}/R{num}' 
c3 = ' sbatch run.sh'
c4 = ' cd - '

print('Executing ...')
os.system( ' ;  '.join([c1,c2,c3,c4]) )
print('Sbatch completed')
