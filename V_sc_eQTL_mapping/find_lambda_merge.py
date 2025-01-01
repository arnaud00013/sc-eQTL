# Here we're going to estimate lambda based on cross validation error across all folds instead of a single fold at a time.

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
import sys
import time
from contextlib import closing
import scipy.io
from scipy.stats import mannwhitneyu, linregress, rankdata, skew
#from scipy.stats import boxcox
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.decomposition import PCA
import pandas as pd

cwd = os.getcwd()
import psutil
process = psutil.Process(os.getpid())

import multiprocessing as mp
from multiprocessing import Pool

#import main workspace absolute path
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[1]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[2]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[3]

#import spore_defs from defined workspace
sys.path.append(workspace_path)
from spore_defs import *

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

# Read SNP map
#SNP_reader = csv.reader(open("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),'r'),delimiter='\t')

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
#exit()

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

from argparse import ArgumentParser, SUPPRESS
# Disable default help
parser = ArgumentParser(add_help=False)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

# Add back help 
optional.add_argument(
    '-h',
    '--help',
    action='help',
    default=SUPPRESS,
    help='show this help message and exit'
)
required.add_argument('-dir', help='Parent directory', required=True)
required.add_argument('-model', help='Whether to find lambda in oCV or in iCV directories. 1 = iCV, 2 = oCV', required=True, type=int)
required.add_argument('-i', help='The file that contains the position and effect sizes to be merged.', required=True)
required.add_argument('--fit', help='Phenotype data / Expression matrix in the case of sc_eQTL')
required.add_argument('--geneid', help='Gene id',type=int)
optional.add_argument('--oCV', help='Outside cross-validation value (k = 0-9)', type=int, default=0)
optional.add_argument('-lambda_output', help='File to output the lambda value.')
optional.add_argument('-krl_output', help='File in KRL style output.')
optional.add_argument('--SE', help='Whether to use a lambda at 1 SE larger.', type=int, default=0)
optional.add_argument('--downsample', help='Number of segregants to downsample.', default=0, type=int)
optional.add_argument('--sporelist', help='Restrict searches to a list of spores.')
optional.add_argument('--unweighted', help='Only run the forward search on unweighted data.', default=0, type=int)

args = parser.parse_args()

# Ok, let's parse the directory structure.
dir_struct = ""
max_range = ""
if(args.model == 2):
	dir_struct = "oCV_"
	max_range = 10
elif(args.model == 1):
	dir_struct = "iCV_"
	max_range = 9
else:
	print("-model must be 1 or 2.", file=sys.stderr)
	exit()


outside_CV = args.oCV # Goes from 0 to 9 # k = 10

if(outside_CV > 9 or outside_CV < 0):
	print("--oCV must be [0,9]")
	exit()


filehandle = sys.stderr
if(args.lambda_output):
	filehandle = open(args.lambda_output,"w")

filehandle_krl = sys.stderr
if(args.krl_output):
	filehandle_krl = open(args.krl_output,"w")

cross_val = []
for i in range(max_range):
	#print(str(args.dir) + "/" + str(dir_struct) + str(i) + "/" + str("cross_val_merge.txt"))
	cross_val.append(np.loadtxt(str(args.dir) + "/" + str(dir_struct) + str(i) + "/" + str("cross_val_merge.txt")))

best_R2 = 0
best_R2_SE = 0
best_threshold = 0
for merge_corr in range(len(cross_val[0])):
	R2s = []
	for i in range(max_range):
		R2 = cross_val[i][merge_corr][1]
		R2s.append(R2)

	R2_average = np.mean(R2s)

	if(R2_average > best_R2):
		best_R2 = R2_average
		best_R2_SE = np.std(R2s)/np.sqrt(len(R2s))
		best_threshold = cross_val[0][merge_corr][0]

thresholds = []
R2_SEs = []
for merge_corr in range(len(cross_val[0])):
	R2s = []
	for i in range(max_range):
		R2 = cross_val[i][merge_corr][1]
		R2s.append(R2)

	R2_average = np.mean(R2s)
	if(R2_average >= best_R2 - best_R2_SE):
		thresholds.append(cross_val[0][merge_corr][0])
		R2_SEs.append(R2_average)

if(args.SE == 1):
	index_SE = np.argmin(thresholds)
	best_R2 = R2_SEs[index_SE]
	best_threshold = thresholds[index_SE]

# Read in the fitness data
mtx_gene_expression = pd.read_csv(args.fit,sep="\t",header=None).to_numpy()
#mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0) #mtx_gene_expression = mtx_gene_expression/np.nansum(a=mtx_gene_expressio>

'''
#Mean expression for barcodes with same reference lineage assignment
mean_E = np.zeros((np.sum(lst_not_yet_duplicated),np.shape(mtx_gene_expression)[1]))
ii_row_non_dupl = 0
for ind_row in np.arange(len(lst_not_yet_duplicated)):
    if lst_not_yet_duplicated[ind_row]:
        current_lin = lst_assigned_lineages[ind_row]
        if (current_lin in lst_already_duplicated):
            lst_indexes_barcodes_with_current_lineage = [i for i, e in enumerate(lst_assigned_lineages) if e == lst_assigned_lineages[ind_row]]
            mean_E[ii_row_non_dupl,:] = np.nanmean(mtx_gene_expression[df_best_matches["pvalue"]<0.05,:][lst_indexes_barcodes_with_current_lineage,:], axis=0)
        else:
            mean_E[ii_row_non_dupl,:] = mtx_gene_expression[df_best_matches["pvalue"]<0.05][ind_row,:]
        ii_row_non_dupl = ii_row_non_dupl + 1
    else:
        continue
'''
fitnesses_data = mtx_gene_expression[:,int(args.geneid)] #mean_E[:,int(args.geneid)] #np.loadtxt(args.fit)
del mtx_gene_expression

# Parse and see if it has standard errors

if(len(fitnesses_data.shape) != 2):
	# No errors found, assume all errors the same.
	if(len(fitnesses_data.shape) == 1):
		fitnesses_data = np.reshape(fitnesses_data,(-1,1))

	fitnesses = fitnesses_data
	#fitnesses = np.reshape(fitnesses,(len(fitnesses_data,1)))
	errors = np.ones(len(fitnesses_data))
else:
	fitnesses = fitnesses_data[:,0]
	errors = fitnesses_data[:,1]

errors = np.square(errors)
errors = np.reciprocal(errors)

seed = 100000
np.random.seed(seed) # This allows us to keep the same cross validation sets.

# First let's take care of the outside CV
sporelist = np.array(range(len(fitnesses)))
if(args.sporelist):
	sporelist = np.loadtxt(args.sporelist, dtype=int)

if(args.downsample > 0 and args.downsample < len(sporelist)):
	#fitnesses = fitnesses[0:args.downsample]
	#errors = errors[0:args.downsample]
	sporelist = sporelist[0:args.downsample]

perm = np.random.permutation(sporelist)


# If model is 2, then we'll only obtain the final coefficients given a known lambda, otherwise cross validate against the test or validation set according to the cross validation.
train_perm = perm.copy()

if(args.model != 2):
	train_perm = np.delete(train_perm, np.r_[outside_CV/10 * len(sporelist):(outside_CV + 1)/10 * len(sporelist)].astype(int),axis=0)

train_set = np.take(fitnesses,train_perm) # If model = 2, then this is the whole set, if model = 1, then this is 90% of the data, and if model = 2 then this is 80% of the data.

train_errors = np.take(errors,train_perm)
train_phenotypes = train_set[~np.isnan(train_set)] # Is a numpy.ndarray
train_errors = train_errors[~np.isnan(train_set)]
train_num_usable_spores = len(train_phenotypes)


# We now have the threshold, so we'll merge the list of QTLs
qtls = np.loadtxt(str(args.dir) + "/" + str(args.i))
best_pos = qtls[0].astype(int)
best_effects = qtls[1]

# Open all the genotype files
genotypes_file = []
num_lines_genotypes = []
chr_to_scan = []
start = time.perf_counter()
i = 0
'''
#create matrix of corrected and imputed genotypes
df_corrected_imputed_genotypes = pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,lst_label_chromosomes[0]),sep="\t",header=None)
for the_label_chr in lst_label_chromosomes[1:16]:
    df_corrected_imputed_genotypes = pd.concat([df_corrected_imputed_genotypes,pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,the_label_chr),sep="\t",header=None)],axis=1)
mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
del df_corrected_imputed_genotypes

#Median genotypes for barcodes with same reference lineage assignment
try:
    median_G = pd.read_csv("{0}/test_combine/median_G.csv".format(workspace_path),sep="\t",header=None).to_numpy()
except:
    median_G = np.zeros((np.sum(lst_not_yet_duplicated),np.shape(mtx_corrected_imputed_genotypes)[1]))
    ii_row_non_dupl = 0
    for ind_row in np.arange(len(lst_not_yet_duplicated)):
        if lst_not_yet_duplicated[ind_row]:
            current_lin = lst_assigned_lineages[ind_row]
            if (current_lin in lst_already_duplicated):
                lst_indexes_barcodes_with_current_lineage = [i for i, e in enumerate(lst_assigned_lineages) if e == lst_assigned_lineages[ind_row]]
                median_G[ii_row_non_dupl,:] = np.nanmedian(mtx_corrected_imputed_genotypes[df_best_matches["pvalue"]<0.05,:][lst_indexes_barcodes_with_current_lineage,:], axis=0)
            else:
                median_G[ii_row_non_dupl,:] = mtx_corrected_imputed_genotypes[df_best_matches["pvalue"]<0.05][ind_row,:]
            ii_row_non_dupl = ii_row_non_dupl + 1
        else:
            continue
'''
max_reads_G = pd.read_csv("{0}/max_reads_G.csv".format(workspace_path),sep="\t",header=None).to_numpy()
for the_label_chr in lst_label_chromosomes[0:len(lst_label_chromosomes)]:
    genotypes_file.append(np.transpose(max_reads_G[:,lst_inds_begin_chrs[i]:(lst_inds_finish_chrs[i]+1)])) #genotypes_file.append(np.transpose(mtx_corrected_imputed_genotypes[:,lst_inds_begin_chrs[i]:(lst_inds_finish_chrs[i]+1)]))
    #print(np.shape(genotypes_file[i])) #sanity check for genotype matrix shape
    num_lines_genotypes.append(genotypes_file[i].shape[0])
    chr_to_scan.append(i)
    print(str(i) + "    " + str(time.perf_counter() - start) + "    " + str(process.memory_info().rss/1024/1024),file=sys.stderr)
    i = i + 1
#del mtx_corrected_imputed_genotypes
#del median_G

X_train = np.ones((train_num_usable_spores,len(best_pos)))


for pos_index in range(len(best_pos)):
	pos = best_pos[pos_index]
	chr_qtl = np.searchsorted(np.array(chrom_startpoints),pos+0.5)
	start_of_chr = chrom_startpoints[chr_qtl-1]
	pos_in_chr = pos - start_of_chr

	pos_line = genotypes_file[chr_qtl-1][pos_in_chr]
	
	train_line = np.take(pos_line, train_perm)
	train_line = train_line[~np.isnan(train_set)]

	X_train[:,pos_index] = train_line.copy()

corr_matrix = np.corrcoef(X_train,rowvar=False)
corr_matrix_tril = np.tril(corr_matrix,k=-1)

corr_groups = np.argwhere(corr_matrix_tril >= best_threshold).tolist()

for i in range(len(corr_groups)):
	j = i+1
	while j < len(corr_groups):
		if(np.in1d(corr_groups[i],corr_groups[j]).any()):
			merge = np.unique(np.concatenate((corr_groups[i],corr_groups[j]),0))
			#np.reshape(corr_groups[i],merge.shape)
			corr_groups[i] = merge
				
			# Delete j
			del corr_groups[j] #corr_groups = np.delete(corr_groups,j)
			# Restart
			j = i
		j = j + 1

# This merging procedure might not always form contiguous elements.
# We need then to merge with more if the positions overlap in space (i.e: [1, 4] and [2, 5] need to merge)
pos_groups = []

for i in range(len(corr_groups)):
	pos_groups.append(best_pos[corr_groups[i]].tolist())


for i in range(len(pos_groups)):
	j = i + 1
	while j < len(pos_groups):
		min_i = np.amin(pos_groups[i])
		min_j = np.amin(pos_groups[j])
		max_i = np.amax(pos_groups[i])
		max_j = np.amax(pos_groups[j])

		# Only merge if things are on the same chromosome
		chr_qtl_min_i = np.searchsorted(np.array(chrom_startpoints),min_i+0.5)
		chr_qtl_max_i = np.searchsorted(np.array(chrom_startpoints),max_i+0.5)
		chr_qtl_min_j = np.searchsorted(np.array(chrom_startpoints),min_j+0.5)
		chr_qtl_max_j = np.searchsorted(np.array(chrom_startpoints),max_j+0.5)

		if(np.in1d(np.arange(min_i,max_i+1),np.arange(min_j,max_j+1)).any() and chr_qtl_min_i == chr_qtl_max_j and chr_qtl_max_i == chr_qtl_min_j):
			merge = np.unique(np.concatenate((pos_groups[i],pos_groups[j]),0))
			pos_groups[i] = merge
			del pos_groups[j] # pos_groups = np.delete(pos_groups,j) # 
			j = i
		j = j + 1

# Now populate an initial beta array with the effects.
# Make a copy of the best_effects array
initial_beta = best_effects.copy()

krl_initial_beta = best_effects.copy()

# Now go through the merges and assign value of initial_beta
count_effective = len(initial_beta)
for i in range(len(pos_groups)):
	sum = 0
	for j in range(len(pos_groups[i])):
		index = np.argwhere(best_pos == pos_groups[i][j])[0][0]
		sum = sum + best_effects[index]

	for j in range(len(pos_groups[i])):
		index = np.argwhere(best_pos == pos_groups[i][j])[0][0]
		initial_beta[index] = sum/len(pos_groups[i])

	count_effective = count_effective - (len(pos_groups[i])-1)

used_index = []
for i in range(len(pos_groups)):
	sum = 0
	for j in range(len(pos_groups[i])):
		index = np.argwhere(best_pos == pos_groups[i][j])[0][0]
		sum = sum + best_effects[index]

	print(sum, end="", file=filehandle_krl)
	for j in range(len(pos_groups[i])):
		index = np.argwhere(best_pos == pos_groups[i][j])[0][0]
		used_index.append(index)
		print("\t" + str(best_pos[index]), end="", file=filehandle_krl)
	print("", file=filehandle_krl)

for i in range(len(best_pos)):
	if(~np.isin(i, used_index)):
		print(str(krl_initial_beta[i]) + "	" + str(best_pos[i]), file=filehandle_krl)

print(str(best_threshold) + "	" + str(best_R2) + "	" + str(len(initial_beta)) + "	" + str(count_effective), file=filehandle)


print(*best_pos)
print(*initial_beta)
	
exit()
