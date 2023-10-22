#!/bin/bash
#SBATCH --account=rrg-shapiro
#SBATCH --nodes=3                 # Number of nodes
#SBATCH --ntasks=100               # Number of MPI process
#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=100G                # memory per node
#SBATCH --time=7-00:00            # time (DD-HH:MM)
python HMM_based_Expression_heritability.py $the_ind_partition $nb_expression_pcs_partitions $nb_expression_PCs $workspace $cellranger_outs_folder $Nb_cpus
