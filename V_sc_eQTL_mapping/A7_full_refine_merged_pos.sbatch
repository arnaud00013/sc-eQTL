#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J refine_pos_all # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 2-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=200G

FOLDER="/home/p1211536/scratch/NoFastp_Yeast"
FITNESS_FILE="pheno_data_30C.txt"

~/Tensorflow_mod/bin/python3.9 refine_pos_all.py -qtl_pos merged_refined_pos.txt --model 2 --dir $FOLDER --fit $FITNESS_FILE --unweighted 1 >$FOLDER/refined_merged.txt
