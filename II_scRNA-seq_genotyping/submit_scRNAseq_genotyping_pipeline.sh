#!/bin/bash

#header

#create repertory for data and output
/bin/rm -rf data
/bin/mkdir data

#paths of the RM and BY workspace
THE_CWD_PATH=`pwd`

#Delete existing cell genotypes file
/bin/rm -rf data/good_cells_genotypes

#Create output repertories
/bin/mkdir data/good_cells_genotypes
/bin/mkdir data/good_cells_genotypes/plots

#Create allele count and allele ratio files for each barcode
python pysam_split.py $THE_CWD_PATH 16

#Create a file with the list of Cells chromosome genotypes
/bin/rm -f data/lst_cells_chrom_genotypes.txt
/bin/touch data/lst_cells_chrom_genotypes.txt
for f in data/good_cells_genotypes/*.csv; do basename -s .csv $f >> data/lst_cells_chrom_genotypes.txt; done

#Run error-correcting HMM
python Github_HMM.py $THE_CWD_PATH $THE_CWD_PATH $Nb_cells $Nb_cpus

#Count the number of reads per barcode
samtools view data/RM_mapped.bam | grep CB:Z: | sed 's/.*CB:Z:\([ACGT]*\).*/\1/' | sort | uniq -c > reads_per_barcode
