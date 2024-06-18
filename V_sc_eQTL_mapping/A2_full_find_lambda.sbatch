#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J find_lambda # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 2-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=200G

FOLDER="/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL"
FITNESS_FILE="max_tot_Reconstructed_E.csv"

~/Tensorflow_mod/bin/python3.9 find_lambda_3.py -dir $FOLDER -model 2 -fit $FITNESS_FILE -lambda_output $FOLDER/lambda.txt >$FOLDER/unrefined.txt
