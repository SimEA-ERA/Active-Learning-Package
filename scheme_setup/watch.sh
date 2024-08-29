#!/bin/bash

squeue -u npatsalidis -p milan >w.out
grep  -E 'R([1-9]|[1-9][0-9]|100)\b' w.out > w1.out
tail -n 4 w1.out
e=evaluations
tail -n 3 $e/mae.dat $e/pred_mae.dat $e/mse.dat $e/pred_mse.dat 
echo '------Watching out -----'
tail -n 6 out

rm w.out w1.out
