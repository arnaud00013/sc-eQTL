#!/bin/bash
#header

cellranger count --id=map2RM_ref --transcriptome=$RM_reference --fastqs=$Path_illumina_reads --sample=$name_sample --localcores=$Nb_cpu --localmem=$memory_allocated