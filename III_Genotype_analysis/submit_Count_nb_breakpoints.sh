#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=16               # Number of MPI process
#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=200G                # memory per node
#SBATCH --time=03-00:00            # time (DD-HH:MM)

#Run error-correcting HMM
python Count_nb_breakpoints_per_HMM_imputations.py $workspace $cellranger_outs_folder $Nb_cpus
python Count_nb_breakpoints_per_reflineage.py $workspace $cellranger_outs_folder $Nb_cpus
