#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=16               # Number of MPI process
#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=200G                # memory per node
#SBATCH --time=03-00:00            # time (DD-HH:MM)

#Run error-correcting HMM
python generate_dist_cell_to_lineage_corrected_gen.py $workspace $cellranger_outs_folder $Nb_cpus $Number_of_subsampes_for_lineage_assignment
