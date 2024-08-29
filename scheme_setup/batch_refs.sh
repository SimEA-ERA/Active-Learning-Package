#!bin/bash
out=$(sbatch run.sh)
jobid=${out##* }

while true; do
squeue -u npatsalidis --job $jobid >report_of_run
line_count=$(wc -l < "report_of_run")
if [[ $line_count -le 1 ]]; then
     break
fi
sleep 6
done


