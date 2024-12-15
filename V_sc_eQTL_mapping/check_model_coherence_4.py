import string
import numpy as np
from scipy import linalg
import sys
import csv
import itertools
import time
import random
import argparse
import os
import re
import lap
import pandas as pd
cwd = os.getcwd()
from collections import defaultdict

import multiprocessing as mp
from multiprocessing import Pool


#import main workspace absolute path
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[1]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[2]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[3]

#import spore_defs from defined workspace
sys.path.append(workspace_path)
from spore_defs import *
from argparse import ArgumentParser, SUPPRESS

#import data
df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_lst_cells = pd.read_csv("{0}/lst_barcodes_with_expression_data.txt".format(workspace_path),sep="\t",header=None,dtype={ '0': str })
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"

#define list of chromosome beginning and finish indexes
the_current_chr = ""
lst_inds_begin_chrs = []
lst_inds_finish_chrs = []
for ind in np.arange(len(df_pos_snps.chromosome.tolist())):
	if the_current_chr != df_pos_snps.chromosome.tolist()[ind]:
		lst_inds_begin_chrs.append(ind)
		if len(lst_inds_begin_chrs) > 1:
			lst_inds_finish_chrs.append(ind-1)
	if ind == (len(df_pos_snps.chromosome.tolist())-1):
		lst_inds_finish_chrs.append(ind)
	the_current_chr = df_pos_snps.chromosome.tolist()[ind]

df_best_matches = pd.read_csv("{0}/backup_expected_distance_df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })

#find duplicates for the assigned reference lineages
lst_assigned_lineages = np.array(df_best_matches[df_best_matches["pvalue"]<0.05].best_match.tolist()).tolist()
lst_not_yet_duplicated = []
lst_already_duplicated = []
for current_lin in lst_assigned_lineages:
	if (lst_assigned_lineages.count(current_lin) > 1):
		if (not current_lin in lst_already_duplicated):
			lst_not_yet_duplicated.append(True)
			lst_already_duplicated.append(current_lin)
		else:
			lst_not_yet_duplicated.append(False)
	else:
		lst_not_yet_duplicated.append(True)

#Combine barcode allele counts for cells from the same lineage
lst_unique_lineages_not_yet_duplicated = [lst_assigned_lineages[the_i] for the_i in np.arange(len(lst_not_yet_duplicated)) if lst_not_yet_duplicated[the_i]]

# Disable default help
parser = ArgumentParser(add_help=False)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

# Add back help 
optional.add_argument('-h','--help',action='help',default=SUPPRESS,help='show this help message and exit')
required.add_argument('-m', help='File that contains the betas', required=True)
required.add_argument('-t', help='File that contains the true betas', required=True)
optional.add_argument('-sd', help='Standardize the coefficients by the standard deviation only', action='store_true')
optional.add_argument('--scratch', help='Local scratch directory', default='/n/holyscratch01/desai_lab/nnguyenba/BBQ/all_data/genomes/')
optional.add_argument('--getdeltas', help='Only print deltas', action='store_true')
optional.add_argument('--randpos', help='Randomize positions', action='store_true')

args = parser.parse_args()

# Read SNP map
#SNP_reader = csv.reader(open('/n/desai_lab/users/klawrence/BBQ/alldata/BYxRM_nanopore_SNPs.txt','r'),delimiter='\t')
#SNP_reader = csv.reader(open('/n/home00/nnguyenba/scripts/BBQ/alldata/BYxRM_nanopore_SNPs.txt','r'),delimiter='\t')
#SNP_reader = csv.reader(open('/n/holyscratch01/desai_lab/nnguyenba/BBQ/all_data/BYxRM_nanopore_SNPs.txt','r'),delimiter='\t')
#genome_str = genome_str_to_int(next(SNP_reader))
#SNP_list = genome_to_chroms(genome_str)
#num_chroms = len(SNP_list)
#num_SNPs = [len(x) for x in SNP_list]
#num_SNPs_total = sum(num_SNPs)
#print(num_SNPs,file=sys.stdout,flush=True)
#print(num_SNPs_total,file=sys.stdout,flush=True)
#chrom_startpoints = get_chrom_startpoints(genome_str)
#chrom_endpoints = get_chrom_endpoints(genome_str)

chrom_startpoints = [0, 996, 4732, 5291, 9327, 11187, 12476, 16408, 18047, 20126, 23101, 26341, 30652, 33598, 35398, 39688]
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41593]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1906]

# Read in the file with positions
model_pos = []
model_beta = []
model_pos = np.array(model_pos)

with open(args.m,'r') as readfile:
	linecount = 0
	for line in readfile:
		line = line.rstrip()
		if(linecount % 4 == 0):
			# positions
			pos = np.array(re.split("[\s\t]",line))
			model_pos = np.take(pos,np.where(np.char.find(pos,",") < 0))[0].astype(int)

			#model_pos = np.fromstring(line, sep="	",dtype=int)

		if(linecount %4 == 1):
			# Effects
			model_beta = np.fromstring(line, sep="	",dtype=float)
		linecount = linecount + 1

# Normalize beta if needed
model_std = np.std(model_beta)
if(args.sd):
	model_beta = model_beta/model_std

# Randomize positions if required
if(args.randpos):
	#model_pos = np.random.permutation(num_SNPs_total)[:model_pos.size]
	model_pos = np.random.permutation(model_pos)


# Now let's work with the positions and the genotypes.
# First, let's get an array of the sorted positions.
model_sorted_indexes = np.argsort(model_pos)
model_sorted_prev_pos = np.take(model_pos,model_sorted_indexes)
model_sorted_beta = np.take(model_beta,model_sorted_indexes)
model_reverse_sorted_indexes = np.argsort(model_sorted_indexes)

model = defaultdict(list)
model_beta = {}

# Populate an array for the snps for each chromosomes obtained
count = 0
for i in model_sorted_prev_pos:
	chr_of_snp = np.searchsorted(np.array(chrom_startpoints),i+0.5)
	model[chr_of_snp-1].append(i)
	model_beta[i] = model_sorted_beta[count]
	count = count + 1

# Do the same for the truth
truth_pos = []
truth_beta = []
truth_pos = np.array(truth_pos)

with open(args.t,'r') as readfile:
	linecount = 0
	for line in readfile:
		line = line.rstrip()
		if(linecount % 4 == 0):
			# positions
			pos = np.array(re.split("[\s\t]",line))
			truth_pos = np.take(pos,np.where(np.char.find(pos,",") < 0))[0].astype(int)

			#truth_pos = np.fromstring(line, sep="	",dtype=int)

		if(linecount %4 == 1):
			# Effects
			truth_beta = np.fromstring(line, sep="	",dtype=float)
		linecount = linecount + 1

truth_std = np.std(truth_beta)
if(args.sd):
	truth_beta = truth_beta/truth_std

if(args.randpos):
	#truth_pos = np.random.permutation(num_SNPs_total)[:truth_pos.size]
	truth_pos = np.random.permutation(truth_pos)

# Now let's work with the positions and the genotypes.
# First, let's get an array of the sorted positions.
truth_sorted_indexes = np.argsort(truth_pos)
truth_sorted_prev_pos = np.take(truth_pos,truth_sorted_indexes)
truth_sorted_beta = np.take(truth_beta,truth_sorted_indexes)
truth_reverse_sorted_indexes = np.argsort(truth_sorted_indexes)

truth = defaultdict(list)
truth_beta = {}

# Populate an array for the snps for each chromosomes obtained
count = 0
for i in truth_sorted_prev_pos:
	chr_of_snp = np.searchsorted(np.array(chrom_startpoints),i+0.5)
	truth[chr_of_snp-1].append(i)
	truth_beta[i] = truth_sorted_beta[count]
	count = count + 1

# For each chromosomes, look at pairwise RSS
total_cost = 0
total_price = 0
total_max_cost = 0
truth_max_cost = 0
total_max_price = 0
model_max_cost = 0

# Get the RSS for all the chromosomes not in the model
#for the_index_chr in truth:
#	if(~np.isin(the_index_chr,list(model.keys()))):
#		genotypes_file = np.load(str(args.scratch) + str("/the_index_chr")+str(the_index_chr+1)+"_pos_major.npy", mmap_mode="r")
#		for pos in truth[the_index_chr]:
#			truth_genome = genotypes_file[pos - (chrom_startpoints[the_index_chr]+1)]
#			TSS_2 = np.sum((0 - truth_genome * truth_beta[pos])**2)
#			truth_max_cost = truth_max_cost + TSS_2
#			#total_cost = total_cost + TSS_2
#deltas = []
m = []
t = []

#print(model)
#print(truth)

#print(np.unique(np.concatenate((list(model.keys()),list(truth.keys())))))
'''
#create matrix of corrected and imputed genotypes
df_corrected_imputed_genotypes = pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,lst_label_chromosomes[0]),sep="\t",header=None)
for the_label_chr in lst_label_chromosomes[1:16]:
    df_corrected_imputed_genotypes = pd.concat([df_corrected_imputed_genotypes,pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,the_label_chr),sep="\t",header=None)],axis=1)
mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
del df_corrected_imputed_genotypes
#Median genotypes for barcodes with same reference lineage assignment
#median_G = pd.read_csv("{0}/test_combine/median_G.csv".format(workspace_path),sep="\t",header=None).to_numpy()
'''
# Open all the genotype files
#model_genotypes_file = []
truth_genotypes_file = []
for the_index_chr in np.arange(len(lst_label_chromosomes)):
	#model_genotypes_file.append(np.transpose(median_G[:,lst_inds_begin_chrs[the_index_chr]:(lst_inds_finish_chrs[the_index_chr]+1)]))
	truth_genotypes_file.append(np.transpose(np.loadtxt(workspace_path+"/{0}_spore_major.txt".format(lst_label_chromosomes[the_index_chr]))))
	model_genome = defaultdict(list)
	truth_genome = defaultdict(list)

	# Now loop through the model and grab the genome lines
	for pos in model[the_index_chr]:
		#print(model_genotypes_file[pos - (chrom_startpoints[the_index_chr]+1)])
		model_genome[pos] = truth_genotypes_file[the_index_chr][pos - (chrom_startpoints[the_index_chr]+1)]

		mean = np.mean(model_genome[pos] * model_beta[pos])
		model_max_cost = model_max_cost + np.sum((model_genome[pos] * model_beta[pos] - mean)**2)

		#print("model" + "	" + str(pos) + "	" + str(model_beta[pos]))


	# Do the same for the truth
	for pos in truth[the_index_chr]:
		truth_genome[pos] = truth_genotypes_file[the_index_chr][pos - (chrom_startpoints[the_index_chr]+1)]
		mean = np.mean(truth_genome[pos] * truth_beta[pos])
		truth_max_cost = truth_max_cost + np.sum((truth_genome[pos] * truth_beta[pos] - mean)**2)

		#print("truth" + "	" +  str(pos) + "	" + str(truth_beta[pos]))


	# Populate a cost and pay matrix
	cost = np.zeros((len(model[the_index_chr])+1, len(truth[the_index_chr])+1))
	
	# We'll append the cost matrix with a single row/col to allow overlap detection needleman-wunch
	# The '0th' column and row are going to be the TSS, while the overlaps are going to be the RSS.
	# "cost" is the truth
	# "price" is the model
	for j in range(len(truth[the_index_chr])):
		truth_pos = truth[the_index_chr][j]
		mean = np.mean(truth_genome[truth_pos] * truth_beta[truth_pos])
		#print(str(mean) + "	" + str(truth_beta[truth_pos]))
		cost[0][j+1] = np.sum((truth_genome[truth_pos] * truth_beta[truth_pos] - mean)**2)
		#cost[0][j+1] = np.sqrt(np.sum((truth_genome[truth_pos] * truth_beta[truth_pos])**2))

	for i in range(len(model[the_index_chr])):
		model_pos = model[the_index_chr][i]
		mean = np.mean(model_genome[model_pos] * model_beta[model_pos])
		cost[i+1][0] = np.sum((model_genome[model_pos] * model_beta[model_pos] - mean)**2)
		#cost[i+1][0] = np.sum((model_genome[model_pos] * model_beta[model_pos])**2)

	for i in range(len(model[the_index_chr])):
		model_pos = model[the_index_chr][i]
		for j in range(len(truth[the_index_chr])):
			truth_pos = truth[the_index_chr][j]
			#mean = np.mean(model_genome[model_pos] * model_beta[model_pos] - truth_genome[truth_pos] * truth_beta[truth_pos])
			mean_model = np.mean(model_genome[model_pos] * model_beta[model_pos])
			mean_truth = np.mean(truth_genome[truth_pos] * truth_beta[truth_pos])

			#RSS = np.sum((model_genome[model_pos] * model_beta[model_pos] - truth_genome[truth_pos] * truth_beta[truth_pos] - mean)**2)
			RSS = np.sum(((model_genome[model_pos] * model_beta[model_pos] - mean_model ) - (truth_genome[truth_pos] * truth_beta[truth_pos] - mean_truth))**2)
			#print(model_beta[model_pos])
			cost[i+1][j+1] = RSS
			#cost[i+1][j+1] = np.sqrt(RSS)		


	price = np.zeros((len(model[the_index_chr])+1, len(truth[the_index_chr])+1))
	traceback_1 = np.zeros((len(model[the_index_chr])+1, len(truth[the_index_chr])+1))
	traceback_2 = np.zeros((len(model[the_index_chr])+1, len(truth[the_index_chr])+1))

	# Fill in the first price row
	# Max cost
	max_cost = 0
	max_price = 0
	for j in range(1,len(truth[the_index_chr])+1):
		price[0][j] = price[0][j-1] + cost[0][j]
		traceback_1[0][j] = 0
		traceback_2[0][j] = j-1
		#max_cost = max_cost + price[0][j]

	# Fill in the first price column
	for i in range(1,len(model[the_index_chr])+1):
		price[i][0] = price[i-1][0] + cost[i][0]
		traceback_1[i][0] = i-1
		traceback_2[i][0] = 0
		#max_price = max_price + price[i][0]

	max_cost = price[0][len(truth[the_index_chr])]
	max_price = price[len(model[the_index_chr])][0]

	# Fill in the price matrix, and store tracebacks
	for i in range(1,len(model[the_index_chr])+1):
		for j in range(1,len(truth[the_index_chr])+1):
			cost_1 = price[i-1][j-1] + cost[i][j] # A match
			cost_2 = price[i-1][j] + cost[i][0] # Better to not match and lose a true QTL
			cost_3 = price[i][j-1] + cost[0][j] # Better to not match and lose a prediction

			min_cost = min(cost_1,cost_2,cost_3)

			if(min_cost == cost_1):
				price[i][j] = cost_1
				traceback_1[i][j] = i-1
				traceback_2[i][j] = j-1
			elif(min_cost == cost_2):
				price[i][j] = cost_2
				traceback_1[i][j] = i-1
				traceback_2[i][j] = j
			else:
				price[i][j] = cost_3
				traceback_1[i][j] = i
				traceback_2[i][j] = j-1


	#print(price)
	#print(cost)

	#print(traceback_1)
	#print(traceback_2)

	# Follow the traceback.
	i = int(len(model[the_index_chr]))
	j = int(len(truth[the_index_chr]))
	# First column printed is the model
	# Second column printed is the truth
	condition = True
	while(condition):
		#print(str(i) + "	" + str(j))
		if(traceback_1[i][j] == 0 and traceback_2[i][j] == 0):
			condition = False

		new_i = traceback_1[i][j]
		new_j = traceback_2[i][j]

		if(new_i == i-1 and new_j == j-1):
			# Match state
			if(not args.getdeltas):
				print(str(model[the_index_chr][i-1]) +  "	" + str(truth[the_index_chr][j-1]) + "	" + str(model_beta[model[the_index_chr][i-1]]) + "	" + str(truth_beta[truth[the_index_chr][j-1]]))
			#deltas.append(abs(model_beta[model[the_index_chr][i-1]] - truth_beta[truth[the_index_chr][j-1]]))
			m.append(model_beta[model[the_index_chr][i-1]])
			t.append(truth_beta[truth[the_index_chr][j-1]])
		elif(new_i == i-1):
			# Model state
			if(not args.getdeltas):
				print(str(model[the_index_chr][i-1]) + "	" + str(model_beta[model[the_index_chr][i-1]]))
			#deltas.append(abs(model_beta[model[the_index_chr][i-1]] - 0))
			m.append(model_beta[model[the_index_chr][i-1]])
			t.append(0)

		elif(new_j == j-1):
			if(not args.getdeltas):
				print("	" + str(truth[the_index_chr][j-1]) + "	" + str(truth_beta[truth[the_index_chr][j-1]]))
			#deltas.append(abs(0 - truth_beta[truth[the_index_chr][j-1]]))
			m.append(0)
			t.append(truth_beta[truth[the_index_chr][j-1]])


		i = int(new_i)
		j = int(new_j)

		#print("new->	" + str(i) + "	" + str(j) + "	" + str(traceback_1[i][j]) + "	" + str(traceback_2[i][j]))
	if(not args.getdeltas):
		print("Price (" + str(the_index_chr) + "):	" + str(price[len(model[the_index_chr])][len(truth[the_index_chr])]) + "	" + "Max cost:	" + str(max_cost) + "	" + "Max price: " + str(max_price))
	total_price = total_price + price[len(model[the_index_chr])][len(truth[the_index_chr])]
	total_max_cost = total_max_cost + max_cost
	total_max_price = total_max_price + max_price

#del mtx_corrected_cell_genotypes

if(args.getdeltas):
	
	m = np.asarray(m)
	t = np.asarray(t)

	deltas = m-t
	deltas = np.absolute(deltas)

	# Sort all the deltas
	deltas = deltas[np.argsort(-deltas)]

	sum_deltas = sum(deltas)

	cumulative_sum = 0
	for i in range(len(deltas)):
		print(str(deltas[i]))

else:
	print(total_price)
	print(total_max_cost)
	print(total_max_price)
	print(1-total_price / (total_max_cost + total_max_price))


exit()
