#!/bin/bash

####################################
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=def  # Job name
#SBATCH --output=def.out # Stdout (%j expands to jobId)
#SBATCH --error=def.err # Stderr (%j expands to jobId)
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --partition=milan
#SBATCH --time=00:53:00   # walltime

nproc=1
## LOAD MODULES ##
#module purge        # clean up loaded modules 
module load LAMMPS/23Jun2022-foss-2021b-kokkos
mpirun -np $nproc lmp -in sample_run.lmscr   >output_lammps 2>error_lammps
#mpirun -np $nproc lmp -in stabil.lmscr
