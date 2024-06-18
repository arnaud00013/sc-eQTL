#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J find_lambda_merge # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 2-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=200G

FOLDER="/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL"
FITNESS_FILE="max_tot_Reconstructed_E.csv"

~/Tensorflow_mod/bin/python3.9 find_lambda_merge.py -dir $FOLDER/gene_id_$the_gene_id -model 2 -i refined.txt -fit $FITNESS_FILE -lambda_output $FOLDER/gene_id_$the_gene_id/lambda_merge_SE.txt -krl_output $FOLDER/gene_id_$the_gene_id/krl_merged_SE.txt --SE 1 >$FOLDER/gene_id_$the_gene_id/merged_SE.txt
