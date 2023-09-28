#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH -J RUN_All_B3_sc_eQTL # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --cpus-per-task 1
#SBATCH -t 2-20:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem=10G

declare -i start_gene_id=0
declare -i end_gene_id=89
declare -i current_uplimit_gene_id=90
declare -i nb_running_processes=0
declare -i limit_nb_proc=1
#declare -i nb_completed_ocvs=0
#declare -i nb_oCVs_to_complete=90
while [ $end_gene_id -lt 900 ]
do
  sbatch --array=$start_gene_id-$end_gene_id RUN_B3_SC_EQTL.sh
  nb_running_processes=$(squeue | nl | grep p1211536 | wc -l)
  while [ "$limit_nb_proc" -lt "$nb_running_processes" ]
  do
    nb_running_processes=$(squeue | nl | grep p1211536 | wc -l)
  done
  #nb_lines_out_B3=$(wc -l lst_out_B3_p2.txt | awk '{print $1}')
  #while [ $nb_lines_out_B3 -lt $current_uplimit_gene_id ]

  #nb_oCVs_to_complete=$(( current_uplimit_gene_id-start_gene_id ))
  #nb_oCVs_to_complete=$(( 10*nb_oCVs_to_complete ))
  #while [ "$nb_completed_ocvs" -lt "$nb_oCVs_to_complete" ]
  #do
  #  sleep 30s
  #  nb_completed_ocvs=$(grep -lP "^$50\t" /home/p1211536/scratch/NoFastp_Yeast/sc_eQTL/gene_id_$end_gene_id/oCV_*/cross_val.txt | wc -l)
  #  printf 'Number of completed out cross-validations for last gene expression PC = %s\n' "$nb_completed_ocvs"
    #ls gene_id*/B2_done.txt > lst_out_B2_p2.txt
    #nb_lines_out_B2=$(wc -l lst_out_B2_p2.txt | awk '{print $1}')
    #printf '%s\n' "$nb_lines_out_B2"
  #done
  echo 'start next round of 90 genes'
  printf 'previous start gene id is %s\n' "$start_gene_id"
  printf 'previous end gene id is %s\n' "$end_gene_id"
  printf 'previous gene id upper limit is %s\n' "$current_uplimit_gene_id"
  start_gene_id=$(( start_gene_id + 90 ))
  end_gene_id=$(( end_gene_id + 90 ))
  current_uplimit_gene_id=$(( end_gene_id + 1 ))
  printf 'current start gene id is %s\n' "$start_gene_id"
  printf 'current end gene id is %s\n' "$end_gene_id"
  printf 'current gene id upper limit is %s\n' "$current_uplimit_gene_id"
done

#rm -f lst_out_B3.txt
#touch lst_out_B3.txt
#declare -i start_gene_id=0
#declare -i end_gene_id=89
#declare -i current_uplimit_gene_id=90
#declare -i nb_end_id_corrections=0
#while [ $end_gene_id -lt 900 ] #& [ $nb_end_id_corrections -lt 2 ]
#do
#  sbatch --array=$start_gene_id-$end_gene_id RUN_B3_SC_EQTL.sh
#  nb_lines_out_B3=$(wc -l lst_out_B3.txt | awk '{print $1}')
#  while [ $nb_lines_out_B3 -lt $current_uplimit_gene_id ]
#  do
#    sleep 5m
#    ls gene_id*/B3_done.txt > lst_out_B3.txt
#    nb_lines_out_B3=$(wc -l lst_out_B3.txt | awk '{print $1}')
#  done
#  start_gene_id=$(( start_gene_id + 90 ))
#  end_gene_id=$(( end_gene_id + 90 ))
#  if [ $end_gene_id ge 6240 ]
#  then
#    end_gene_id=6239
#    nb_end_id_corrections=$(( nb_end_id_corrections + 1 ))
#  fi
#  current_uplimit_gene_id=$(( end_gene_id + 1 ))
#done
