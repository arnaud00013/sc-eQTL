#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH -J batch1_oCV_full_QTL # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 1-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=80G
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

FOLDER="/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL/gene_id_$the_gene_id/oCV_$SLURM_ARRAY_TASK_ID"
FITNESS_FILE="max_tot_Reconstructed_E.csv"

~/Tensorflow_mod/bin/python3.9 step_QTL.py --model 1 --dir $FOLDER --fit $FITNESS_FILE --oCV $SLURM_ARRAY_TASK_ID --maxqtl 300 --unweighted 1 --nosave 1 --cpu 16
