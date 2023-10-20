#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=16               # Number of MPI process
#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=200G                # memory per node
#SBATCH --time=03-00:00            # time (DD-HH:MM)

#Run error-correcting HMM
python generate_dist_cell_to_lineage_uncorrected_gen.py /home/p1211536/scratch/NoFastp_Yeast/test_param_bbqtl_hmm_on_scrna_seq /home/p1211536/scratch/NoFastp_Yeast/test_param_bbqtl_hmm_on_scrna_seq /home/p1211536/scratch/NoFastp_Yeast/test_param_bbqtl_hmm_on_scrna_seq 16 500
