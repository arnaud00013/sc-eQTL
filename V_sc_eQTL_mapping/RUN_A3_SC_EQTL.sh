#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J RUN_A3_sc_eQTL_yeast_genes # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 1
#SBATCH -t 0-00:05 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=1G

#mapfile -t the_list_genes < /home/p1211536/scratch/NoFastp_Yeast/sc_eQTL/lst_genes_with_expression_variability.csv
#the_gene=${the_list_genes[$SLURM_ARRAY_TASK_ID]}
the_gene=$SLURM_ARRAY_TASK_ID
sbatch --export=the_gene_id=$the_gene A3_full_refine_pos_unweighted.sbatch
