#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <datapath> <inff> <charge_map> <mass_map>"
    exit 1
fi

# Assign command-line arguments to variables
datapath="$1"
inff="$2"
charge_map="$3"
mass_map="$4"

wp=lammps_working
files=("$datapath"/*.xyz)
 > "$wp/PotEng.xyz"

 for file in "${files[@]}"; do
	echo $file
	datafile="$file"
	python setup_lammps_force_calculation.py -df $datafile -wp $wp -in $inff  -cm ${charge_map} -mm ${mass_map} 
	cd $wp
	bash eval_lammps_force.sh
	python extract_energies.py
	wait
	cd -
	python read_lammps.py -wp $wp -l $(echo "$datafile" | sed -e 's/\.xyz/\.log/g' -e 's/D/L/g')
done
#cp results/ff
#bash make_lammps_file.sh
#for nld in 0 1 2 3;
#do
#    bash make_potential_file.sh
#    run lammps
#    python compare_ffdevelop_lammps_predictions.py
#done
