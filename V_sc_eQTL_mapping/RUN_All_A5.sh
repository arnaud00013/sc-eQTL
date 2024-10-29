#!/bin/bash
#SBATCH --account=def-anguyen
#SBATCH -J RUN_All_A5_sc_eQTL # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 1
#SBATCH -t 0-01:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=10G

declare -i start_gene_id=0
declare -i end_gene_id=899
sbatch --array=$start_gene_id-$end_gene_id RUN_A5_SC_EQTL.sh

#rm -f lst_out_A5.txt
#touch lst_out_A5.txt
#declare -i start_gene_id=0
#declare -i end_gene_id=899
#declare -i current_uplimit_gene_id=900
#declare -i nb_end_id_corrections=0
#while [ $end_gene_id -lt 900 ] #& [ $nb_end_id_corrections -lt 2 ]
#do
#  sbatch --array=$start_gene_id-$end_gene_id RUN_A5_SC_EQTL.sh
#  nb_lines_out_A5=$(wc -l lst_out_A5.txt | awk '{print $1}')
#  while [ $nb_lines_out_A5 -lt $current_uplimit_gene_id ]
#  do
#    sleep 5m
#    ls gene_id*/A5_done.txt > lst_out_A5.txt
#    nb_lines_out_A5=$(wc -l lst_out_A5.txt | awk '{print $1}')
#  done
#  start_gene_id=$(( start_gene_id + 900 ))
#  end_gene_id=$(( end_gene_id + 900 ))
#  if [ $end_gene_id ge 6240 ]
#  then
#    end_gene_id=6239
#    nb_end_id_corrections=$(( nb_end_id_corrections + 1 ))
#  fi
#  current_uplimit_gene_id=$(( end_gene_id + 1 ))
#done
