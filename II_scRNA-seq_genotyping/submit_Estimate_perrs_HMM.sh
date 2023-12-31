#!/bin/bash
#HEADER

#create repertory for data and output
/bin/rm -rf data
/bin/mkdir data

#paths of the RM and BY workspace
RM_ref_file="RM11a.fa"
THE_CWD_PATH=`pwd` 
PATH1=$PATH_TO_RM_BAM_FILE_REPOSITORY
PATH2=$PATH_TO_BY_BAM_FILE_REPOSITORY
PATH_TO_PYTHON_SCRIPTS=$PATH_TO_PYTHON_SCRIPTS

#Focus on mapped reads for both references
samtools view -b -F 4 $PATH1/possorted_genome_bam.bam > data/RM_mapped.bam
samtools view -b -F 4 $PATH2/possorted_genome_bam.bam > data/BY_mapped.bam

#Get header 
samtools view -H data/RM_mapped.bam | grep @SQ > data/headerTest.txt

#list of barcodes with expression data
/bin/zcat $PATH1/filtered_feature_bc_matrix/barcodes.tsv.gz > lst_barcodes_with_expression_data.txt
mapfile -t lst_barcodes_with_expression_data < lst_barcodes_with_expression_data.txt

#sort and index multicell bams
samtools index $THE_CWD_PATH/data/mapped_and_filtered_RM_reads.bam
samtools sort $THE_CWD_PATH/data/RM_mapped.bam > $THE_CWD_PATH/data/sorted_RM_mapped.bam
mv $THE_CWD_PATH/data/sorted_RM_mapped.bam $THE_CWD_PATH/data/RM_mapped.bam
samtools sort $THE_CWD_PATH/data/BY_mapped.bam > $THE_CWD_PATH/data/sorted_BY_mapped.bam
mv $THE_CWD_PATH/data/sorted_BY_mapped.bam $THE_CWD_PATH/data/BY_mapped.bam
samtools index $THE_CWD_PATH/data/RM_mapped.bam
samtools index $THE_CWD_PATH/data/BY_mapped.bam

python Estimate_perrs_HMM.py *$workspace_path $bam_filename $list_of_barcodes_filename $number_of_minimum_mismatch_within_the_same_read_for_index_swapping $minimum_coverage_per_site $number_of_cpus
