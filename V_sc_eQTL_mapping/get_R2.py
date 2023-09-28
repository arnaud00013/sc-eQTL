# So we've identified QTLs through a forward search.
# This forward search already identified a number lambda that maximizes the predictive performance of the model.
# However, we believe that the values of the coefficients are biased away from zero. In some cases, where the predictors are strongly away from zero and in opposition, the penalty for two parameters is less than the penalty for extreme values. 
# Unfortunately, we believe that selection coefficients strongly away from 0 are unlikely, so shrinkage is still required.
# So we're going to take the forward search solution, and apply shrinkage, by lasso, ridge, or elastic net, whatever is easier to implement.

# So first, let's grab the positions of the forward search solution.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41608]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1921]
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
required.add_argument('-i', help='File that contains the positions', required=True) # This file is a single row containing all the positions we are considering.
required.add_argument('--fit', help='Phenotype data / Expression matrix in the case of sc_eQTL')
required.add_argument('--geneid', help='Gene id',type=int)
optional.add_argument('--oCV', help='Outside cross-validation value (k = 0-9)', type=int, default=0)
optional.add_argument('--iCV', help='Inside cross-validation value (l = 0-8)', type=int, default=0)
optional.add_argument('--model', help='Whether to fit on the training set (m = 0), on the train+test set (m = 1) or on the complete data (m = 2)', type=int, default=0)
optional.add_argument('--unweighted', help='Only run the forward search on unweighted data.', default=0, type=int)
optional.add_argument('--downsample', help='Number of segregants to downsample.', default=0, type=int)
optional.add_argument('--sporelist', help='Restrict searches to a list of spores.')

args = parser.parse_args()


outside_CV = args.oCV # Goes from 0 to 9 # k = 10
inside_CV = args.iCV # Goes from 0 to 8 # l = 9

if(outside_CV > 9 or outside_CV < 0):
	print("--oCV must be [0,9]")
	exit()

if(inside_CV > 8 or inside_CV < 0):
	print("--iCV must be [0,8]")
	exit()

if(~np.isin(args.model , range(3))):
	print("--model must be [0,2]")
	exit()


positions = []
effects = []

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

if(len(fitnesses_data.shape) != 2 or args.unweighted == 1):
	# No errors found, assume all errors the same.
	if(len(fitnesses_data.shape) == 1):
		fitnesses_data = np.reshape(fitnesses_data,(-1,1))

	fitnesses = fitnesses_data[:,0]
	#fitnesses = np.reshape(fitnesses,(len(fitnesses_data,1)))
	errors = np.ones(len(fitnesses_data))
else:
	fitnesses = fitnesses_data[:,0]
	errors = fitnesses_data[:,1]

errors = np.square(errors)
errors = np.reciprocal(errors)

seed = 100000
np.random.seed(seed) # This allows us to keep the same cross validation sets.

# Check model against specific spore list.
sporelist = np.array(range(len(fitnesses)))
if(args.sporelist):
	sporelist = np.loadtxt(args.sporelist, dtype=int)

if(args.downsample > 0 and args.downsample < len(sporelist)):
	#fitnesses = fitnesses[0:args.downsample]
	#errors = errors[0:args.downsample]
	sporelist = sporelist[0:args.downsample]


# First let's take care of the outside CV
perm = np.random.permutation(sporelist)


# If model is 2, then we'll only obtain the final coefficients given a known lambda, otherwise cross validate against the test or validation set according to the cross validation.
train_perm = perm.copy()
test_perm = perm.copy()
if(args.model != 2):
	train_perm = np.delete(train_perm, np.r_[outside_CV/10 * len(sporelist):(outside_CV + 1)/10 * len(sporelist)].astype(int),axis=0)
	validation_perm = np.take(perm, np.r_[outside_CV/10 * len(sporelist):(outside_CV + 1)/10 * len(sporelist)].astype(int))
	test_perm = validation_perm

	if(args.model != 1):
		# Ok now let's take care of the inside CV
		# To do this, we split the train_perm into a train/test permutation
		test_perm = np.take(train_perm, np.r_[inside_CV/9 * len(train_perm):(inside_CV + 1)/9 * len(train_perm)].astype(int))
		train_perm = np.delete(train_perm, np.r_[inside_CV/9 * len(train_perm):(inside_CV + 1)/9 * len(train_perm)].astype(int))

# If model is 2, then we'll only run Lasso (reg = linear_model.Lasso(alpha=0.1))
# else, we'll provide a train/test set and use the lassoCV function. Although, I'm not sure that lassoCV can handle our weird way of doing things... like having weighted train, unweighted test. Having an intercept during the train, but also adjusting one for the prediction.
# I think the best way is for us to code the subroutine ourselves.

train_set = np.take(fitnesses,train_perm) # If model = 2, then this is the whole set, if model = 1, then this is 90% of the data, and if model = 2 then this is 80% of the data.
test_set = np.take(fitnesses,test_perm) # If model = 2, then this is the whole set (we won't use it), if model = 1, then this is 10% of the data. If model is 0, then this is also 10% of the data, but another one.

train_errors = np.take(errors,train_perm)
train_phenotypes = train_set[~np.isnan(train_set)] # Is a numpy.ndarray
train_errors = train_errors[~np.isnan(train_set)]
train_num_usable_spores = len(train_phenotypes)
mean_train_phenotypes = np.mean(train_phenotypes)

test_errors = np.take(errors,test_perm)
test_phenotypes = test_set[~np.isnan(test_set)]
test_errors = test_errors[~np.isnan(test_set)]
test_num_usable_spores = len(test_phenotypes)
mean_test_phenotypes = np.mean(test_phenotypes)

TSS = np.sum((test_phenotypes - mean_test_phenotypes)**2)
TSS_2 = np.sum((train_phenotypes - mean_train_phenotypes)**2)

# Ok we have the positions. Now let's grab all the genetic values at these positions.
# This grabs all the genotypes, train and test genotypes.
# Open all the genotype files
genotypes_file = []
num_lines_genotypes = []
chr_to_scan = []
start = time.perf_counter()
i = 0
#create matrix of corrected and imputed genotypes
df_corrected_imputed_genotypes = pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,lst_label_chromosomes[0]),sep="\t",header=None)
for the_label_chr in lst_label_chromosomes[1:16]:
    df_corrected_imputed_genotypes = pd.concat([df_corrected_imputed_genotypes,pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,the_label_chr),sep="\t",header=None)],axis=1)
mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
del df_corrected_imputed_genotypes
'''
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
for the_label_chr in lst_label_chromosomes[0:len(lst_label_chromosomes)]:
    genotypes_file.append(np.transpose(mtx_corrected_imputed_genotypes[:,lst_inds_begin_chrs[i]:(lst_inds_finish_chrs[i]+1)])) #genotypes_file.append(np.transpose(median_G[:,lst_inds_begin_chrs[i]:(lst_inds_finish_chrs[i]+1)]))
    #print(np.shape(genotypes_file[i])) #sanity check for genotype matrix shape
    num_lines_genotypes.append(genotypes_file[i].shape[0])
    chr_to_scan.append(i)
    print(str(i) + "    " + str(time.perf_counter() - start) + "    " + str(process.memory_info().rss/1024/1024),file=sys.stderr)
    i = i + 1
del mtx_corrected_imputed_genotypes
#del median_G

# Now we want to fill a matrix of X. We'll also concatenate the array of phenotypes as the Y.
# Now remember, we want to train on weighted data, but optimize lambda based on unweighted data.
# So, we'll multiply the train genotypes/phenotypes, but leave the test ones alone.

# First create the genotype numpy matrix.
X_test = np.ones((test_num_usable_spores,len(positions)))
X_train = np.ones((train_num_usable_spores,len(positions)))

X_train_2 = np.ones((train_num_usable_spores,len(positions)+1))

for pos_index in range(len(positions)):
	pos = positions[pos_index]
	chr_qtl = np.searchsorted(np.array(chrom_startpoints),pos+0.5)
	start_of_chr = chrom_startpoints[chr_qtl-1]
	pos_in_chr = pos - start_of_chr

	pos_line = genotypes_file[chr_qtl-1][pos_in_chr]
	test_line = np.take(pos_line,test_perm)
	test_line = test_line[~np.isnan(test_set)]

	train_line = np.take(pos_line,train_perm)
	train_line = train_line[~np.isnan(train_set)]

	X_test[:,pos_index] = test_line.copy()
	X_train[:,pos_index] = train_line.copy()

	X_train_2[:,pos_index] = train_line * np.sqrt(train_errors)

X_train_2[:,len(positions)] = X_train_2[:,len(positions)] * np.sqrt(train_errors)

q,r = np.linalg.qr(X_train_2)
beta = linalg.solve_triangular(r,np.dot(np.transpose(q), train_phenotypes * np.sqrt(train_errors)), check_finite=False)
initial_beta_2 = beta[0:len(positions)]

initial_beta = effects

#print(initial_beta)
#print(initial_beta_2)

predicted_phenotypes = np.dot(X_test, initial_beta)
RSS = np.sum((test_phenotypes - np.mean(test_phenotypes) - predicted_phenotypes + np.mean(predicted_phenotypes))**2)
R2 = 1-RSS/TSS

predicted_phenotypes_2 = np.dot(X_train, initial_beta)
RSS_2 = np.sum((train_phenotypes - np.mean(train_phenotypes) - predicted_phenotypes_2 + np.mean(predicted_phenotypes_2))**2)
R2_2 = 1-RSS_2/TSS_2


# Get initial R2.

print(str(R2) + "	" + str(RSS) + "	" + str(TSS) + "	" + str(R2_2) + "	" + str(RSS_2) + "	" + str(TSS_2))

print(np.mean(test_phenotypes))
print(np.mean(predicted_phenotypes_2))
print(predicted_phenotypes)

exit()
