#!/bin/bash
#SBATCH --job-name=Ag7
#SBATCH --output=out  
#SBATCH --error=err  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=32
#SBATCH --partition=milan  
#SBATCH --time=23:59:00  


module load matplotlib/3.4.3-foss-2021b
module load numba/0.54.1-foss-2021b


####variables
datapath="data"
num=46
nextnum=47
charge_map="S:0.4702,O:-0.2351,Ag:0"
mass_map="S:32.065,O:15.999,Ag:107.8682"

results_path=Results
sampling_method=md
nld=1

bash lammps_eval_forces.sh "$datapath/D$nextnum" $results_path/$num/runned.in $charge_map $mass_map 
wait 
if [ $? -ne 0 ]; then
      echo "Error: Evaluating forces did not execute successfully, for num = $num and nld = $nld"
      exit 1
fi
