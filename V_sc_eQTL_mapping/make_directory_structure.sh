#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH -J make_dir_str # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 4
#SBATCH -t 0-5:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=10G

# Create the directory structure
#for id_G in {0..898} #The first 898 PCs of gene expression explain 99% of the expression variance
#do
#    mkdir gene_expr_PCA_id_$id_G
#    cd gene_expr_PCA_id_$id_G
#    for oCV in {0..9}
#    do
#        mkdir oCV_$oCV
#        cd oCV_$oCV
#        for iCV in {0..8}
#        do
#            mkdir iCV_$iCV
#        done
#        cd ..
#    done
#    cd ..
#done

# Create the directory structure
for id_G in {0..6239} #0..6239
do
    mkdir gene_id_$id_G
#    cd gene_id_$id_G
#    for oCV in {0..9}
#    do
#        mkdir oCV_$oCV
#        cd oCV_$oCV
#        for iCV in {0..8}
#        do
#            mkdir iCV_$iCV
#        done
#        cd ..
#    done
#    cd ..
done
