#!/bin/bash
#SBATCH --account=rrg-shapiro
#SBATCH --nodes=3                 # Number of nodes
#SBATCH --ntasks=100               # Number of MPI process
#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=100G                # memory per node
#SBATCH --time=7-00:00            # time (DD-HH:MM)
/home/p1211536/Tensorflow_mod/bin/python3.9 HMM_based_Expression_heritability.py
