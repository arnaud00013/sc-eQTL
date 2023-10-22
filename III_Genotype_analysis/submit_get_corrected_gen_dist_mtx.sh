#!/bin/bash
#HEADER

#Run error-correcting HMM
python generate_dist_cell_to_lineage_corrected_gen.py $workspace $cellranger_outs_folder $Nb_cpus $Number_of_subsampes_for_lineage_assignment
