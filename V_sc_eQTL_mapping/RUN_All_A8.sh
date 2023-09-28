#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J RUN_All_A8_sc_eQTL # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 1
#SBATCH -t 02-20:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=10G

declare -i start_gene_id=0
declare -i end_gene_id=899
declare -i projected_end_gene_id=899
declare -i current_uplimit_gene_id=900
declare -i nb_running_processes=0
declare -i limit_nb_proc=1
#declare -i nb_completed_ocvs=0
#declare -i nb_oCVs_to_complete=90
while [ $end_gene_id -lt 6240 ]
do
  sbatch --array=$start_gene_id-$end_gene_id RUN_A8_SC_EQTL.sh
  nb_running_processes=$(squeue | nl | grep p1211536 | wc -l)
  while [ "$limit_nb_proc" -lt "$nb_running_processes" ]
  do
    nb_running_processes=$(squeue | nl | grep p1211536 | wc -l)
  done
  echo 'start next round of 900 genes'
  printf 'previous start gene id is %s\n' "$start_gene_id"
  printf 'previous end gene id is %s\n' "$end_gene_id"
  printf 'previous gene id upper limit is %s\n' "$current_uplimit_gene_id"
  start_gene_id=$(( start_gene_id + 900 ))
  projected_end_gene_id=$(( end_gene_id + 900 ))
  if [ $projected_end_gene_id -ge 6240 ]
  then
    end_gene_id=6239
  else
    end_gene_id=$(( end_gene_id + 900 ))
  fi
  current_uplimit_gene_id=$(( end_gene_id + 1 ))
  printf 'current start gene id is %s\n' "$start_gene_id"
  printf 'current end gene id is %s\n' "$end_gene_id"
  printf 'current gene id upper limit is %s\n' "$current_uplimit_gene_id"
done
