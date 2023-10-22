#!/bin/bash
#HEADER

#Run script
python generate_pca_expression_partitions.py $nb_expression_pcs_partitions $workspace_path $cellranger_outs_folder $nb_cpus $nb_cells
