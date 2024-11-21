with open('log.lammps','r') as f:
    lines = f.readlines()
    f.closed
for j,line in enumerate(lines):
    if 'Step' and 'PotEng' in line:
        jstep = j
        for i,k in enumerate(line.strip().split()):
            if k=='PotEng':
                ipot = i
pot_en = lines[jstep+1].strip().strip().split()[ipot]
with open('structure.dat','r') as f:
    line = f.readline()
    line = f.readline()
    d = line.strip().split('[')[-1].strip(']').split(',')
    edft = d[0].split('=')[-1]
    ffpot = d[1].split('=')[-1]


print(edft,ffpot,pot_en)
with open('PotEng.xyz','a') as f:
    f.write('EDFT {:s} FFdev {:s} LAMMPS {:s}\n'.format(edft,ffpot,pot_en))
    f.closed
