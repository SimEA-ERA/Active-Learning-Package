#!/bin/bash
datapath="data/"
echo "Please enter a number:"
read number

# Check if the input is a number
if ! [[ "$number" =~ ^[0-9]+$ ]]; then
    echo "Error: Input is not a valid number."
    exit 1
fi

mv ${datapath}/L0 "."
for (( i=$number; i<=30; i++ ))
do
rm -r ${datapath}/*$i
rm -r Results/*$i
rm -r lammps_working/iter$i*
done

mv L0 ${datapath}/
rm lammps_working/potential.inc
rm lammps_working/rigid_fixes.inc
rm lammps_working/*ld
rm lammps_working/log.lammps
rm lammps_working/*txt
rm lammps_working/*lammpstrj
rm lammps_working/output_lammps
rm lammps_working/error_lammps
rm lammps_working/*dat
rm lammps_working/*xyz
rm lammps_working/*tab
if [[ $number -eq 0 ]]; then
    rm evaluations/*dat
fi
rm out err FF_develop.log report_of_run beta_sampling_value
echo "Do you want to delete also the csv and png data ? (If so type 'yes')"
read answer

if [ "$answer" = "yes" ]; then
    # Your code for the next section goes here
    rm  COSTS.csv predictCOSTS.csv
    rm *png
    rm Results/*png
fi
