#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH -J position_with_merging # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 16
#SBATCH -t 2-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=200G

FOLDER="/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL"
FITNESS_FILE="max_tot_Reconstructed_E.csv"

K_param=$(cut -f1 $FOLDER/lambda_merge_SE.txt)

~/Tensorflow_mod/bin/python3.9 position_with_merging.py -qtl_pos refined.txt --fit $FITNESS_FILE --dir $FOLDER/gene_id_$the_gene_id --model 2 --merge $K_param >$FOLDER/gene_id_$the_gene_id/merged_refined_pos.txt

