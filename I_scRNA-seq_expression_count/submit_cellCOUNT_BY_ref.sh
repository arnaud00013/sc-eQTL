#!/bin/bash
#HEADER

cellranger count --id=map2BY_ref --transcriptome=$BY_reference --fastqs=$Path_illumina_reads --sample=$BBQscRNA_1 --localcores=$Nb_cpu --localmem=$memory_allocated
