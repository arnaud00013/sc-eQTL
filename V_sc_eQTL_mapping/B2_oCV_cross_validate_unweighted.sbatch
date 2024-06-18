#!/bin/bash
#SBATCH --account=ctb-bourqueg-ab
#SBATCH -J batch1_oCV_cross_val # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 0-02:20 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=50G
#SBATCH --array=0-9

oCVs=()
oCVs+=('0')
oCVs+=('1')
oCVs+=('2')
oCVs+=('3')
oCVs+=('4')
oCVs+=('5')
oCVs+=('6')
oCVs+=('7')
oCVs+=('8')
oCVs+=('9')

FOLDER="/home/p1211536/scratch/NoFastp_Yeast/batch1_QTL/oCV_$SLURM_ARRAY_TASK_ID"
FITNESS_FILE="batch1_pheno_data_30C.txt"

~/Tensorflow_mod/bin/python3.9 cross_validate_2.py -i $FOLDER/output.txt --model 1 --oCV $SLURM_ARRAY_TASK_ID --fit $FITNESS_FILE >$FOLDER/cross_val.txt
