#!/bin/bash
#SBATCH --job-name=Ag3co2
#SBATCH --output=out  
#SBATCH --error=err  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=16
#SBATCH --partition=milan  
#SBATCH --time=23:59:00  


module load matplotlib/3.4.3-foss-2021b
module load numba/0.54.1-foss-2021b

####variables
datapath="data"
inff=AgCO2.in
bsize=100
Niters=50
iexist=27
contin=27
sigma=0.02
charge_map="C:0.8,O:-0.4,Ag:0"
mass_map="C:12.011,O:15.999,Ag:107.8682"

results_path=Results
sampling_method=md

if [ "$contin" -eq 0 ]; then
 > evaluations/mae.dat
 > evaluations/mse.dat
 > evaluations/elasticnet.dat
 > evaluations/pred_elasticnet.dat
 > evaluations/pred_mae.dat
 > evaluations/pred_mse.dat
fi

for ((num=$contin; num<=$Niters; num++)); do
   if [ "$num" -le 5 ]; then
      echo "sampling method is set to perturbation"
      sampling_method=perturbation
   else
	#if (( num % 1 == 0 )); then
    	#	sampling_method=md
	#else
      echo "sampling method is set to md via lammps"
      sampling_method=md #perturbation
	#fi
   fi
   python active_learning_scheme.py -n $num -dp $datapath -f $inff -b $bsize -s $sigma -exd $iexist -m $sampling_method -cm ${charge_map} -mm ${mass_map}
	if [ $? -ne 0 ]; then
	    echo "Error: fitting algorithm did not execute successfully."
	    exit 1
	fi

	nextnum=$((num + 1))

	if (( nextnum > iexist )); then
	    bash make_rundirs.sh $nextnum $datapath
	    
	    cd "${datapath}/R${nextnum}"
	    pwd
	    out=$(sbatch run.sh)
	    jobid=${out##* }
	    
	    cd -
	    while true; do
		squeue -u npatsalidis --job $jobid >report_of_run
		line_count=$(wc -l < "report_of_run")
		if [[ $line_count -le 1 ]]; then
		     break
		fi
		sleep 10
	    done
	    bash extract_logfiles.sh $nextnum $datapath
	    rm report_of_run
	else
	    echo "Iteration $nextnum not performing DFT since data already exist"
	fi

	inff_eval=$results_path/$num/runned.in
	python evaluate_predictions.py -n $nextnum -f $inff_eval -dp $datapath

	if [ $? -ne 0 ]; then
	    echo "Error: Evaluating algorithm did not execute successfully."
	    exit 1
	fi
	ninit=$(python get_init.py)
	inff=$results_path/$ninit/runned.in

done
