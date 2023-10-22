#!/bin/bash
#HEADER

#Run error-correcting HMM
python Count_nb_breakpoints_per_HMM_imputations.py $workspace $cellranger_outs_folder $Nb_cpus
python Count_nb_breakpoints_per_reflineage.py $workspace $cellranger_outs_folder $Nb_cpus
