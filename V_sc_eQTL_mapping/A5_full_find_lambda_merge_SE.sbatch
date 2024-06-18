#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J find_lambda_merge # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 2-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=200G

FOLDER="/home/p1211536/scratch/NoFastp_Yeast"
FITNESS_FILE="pheno_data_30C.txt"

~/Tensorflow_mod/bin/python3.9 find_lambda_merge.py -dir $FOLDER -model 2 -i refined.txt -fit $FITNESS_FILE -lambda_output $FOLDER/lambda_merge_SE.txt -krl_output $FOLDER/krl_merged_SE.txt --SE 1 >$FOLDER/merged_SE.txt
