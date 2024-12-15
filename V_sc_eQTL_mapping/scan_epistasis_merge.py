# Ok so let's try to code up ridge estimates from the forward search using cross validation.
# So the optimal k has already been found by the find_lambda_2.py script.
# We'll get the position at this lambda for this cross validation loop, and produce a second 'cross_val.txt' file with increasing ridge lambda now.

import string
import numpy as np
from scipy import linalg
import sys
import csv
import itertools
import time
import argparse
import os
cwd = os.getcwd()
import psutil
process = psutil.Process(os.getpid())


#import main workspace absolute path
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[1]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[2]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[3]

sys.path.append(workspace_path)
from spore_defs import *

# Read SNP map
#SNP_reader = csv.reader(open('/n/home00/nnguyenba/scripts/BBQ/alldata/BYxRM_nanopore_SNPs.txt','r'),delimiter='\t')
#genome_str = genome_str_to_int(next(SNP_reader))
#SNP_list = genome_to_chroms(genome_str)
#num_chroms = len(SNP_list)
#num_SNPs = [len(x) for x in SNP_list]
#num_SNPs_total = sum(num_SNPs)
#print(num_SNPs,file=sys.stdout,flush=True)
#print(num_SNPs_total,file=sys.stdout,flush=True)
chrom_startpoints = [0, 996, 4732, 5291, 9327, 11187, 12476, 16408, 18047, 20126, 23101, 26341, 30652, 33598, 35398, 39688]
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41593]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1906]

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
required.add_argument('-i', help='File that contains the forward search output', required=True) # This file is a single row containing all the positions we are considering.
required.add_argument('--fit', help='Phenotype data / Expression matrix in the case of sc_eQTL')
required.add_argument('--geneid', help='Gene id',type=int)
required.add_argument('-k', help='Value of the L0 regularization parameter for the epistasis terms', default=0, type=float)
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


single_positions = []
epistasis_positions = []
single_effects = []
epistasis_effects = []
AICs = []
with open(args.i,'r') as readfile:

	linecount = 0
	for line in readfile:
		line = line.rstrip()
		if(line == "Done"):
			continue
		if(linecount % 4 == 0):
			AIC = float(line)
			AICs.append(AIC)
		
		if(linecount % 4 == 1):
			# positions
			pos = np.array(line.split("	"))
			# Now figure out the epistatic terms.
			epistasis_index = np.argwhere(np.char.find(pos,",")!=-1)[0][0]

			single_positions.append(pos[0:epistasis_index])
			epistasis_positions.append(pos[epistasis_index:len(pos)])

		if(linecount %4 == 2):
			# Effects
			eff = np.fromstring(line, sep="	")
			#effects.append(eff)
			single_effects.append(eff[0:epistasis_index])
			epistasis_effects.append(eff[epistasis_index:len(eff)])

		linecount = linecount + 1

min_likelihood = np.Infinity
best_epistasis_positions = []
best_single_positions = []
best_single_effects = []
best_epistasis_effects = []
for i in range(len(AICs)):
	eps = epistasis_positions[i]
	likelihood = AICs[i] + len(eps) * args.k

	if(likelihood < min_likelihood):
		best_epistasis_positions = eps
		best_single_positions = single_positions[i]
		best_single_effects = single_effects[i]
		best_epistasis_effects = epistasis_effects[i]
		min_likelihood = likelihood


# We now have the best set of position at this regularization parameter.

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

sporelist = np.array(range(len(fitnesses)))
if(args.sporelist):
	sporelist = np.loadtxt(args.sporelist, dtype=int)

if(args.downsample > 0 and args.downsample < len(sporelist)):
	#fitnesses = fitnesses[0:args.downsample]
	#errors = errors[0:args.downsample]
	sporelist = sporelist[0:args.downsample]


# First let's take care of the outside CV
perm = np.random.permutation(len(sporelist))

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

test_errors = np.take(errors,test_perm)
test_phenotypes = test_set[~np.isnan(test_set)]
test_errors = test_errors[~np.isnan(test_set)]
test_num_usable_spores = len(test_phenotypes)
mean_test_phenotypes = np.mean(test_phenotypes)

TSS = np.sum((test_phenotypes - mean_test_phenotypes)**2)

# Ok we have the positions. Now let's grab all the genetic values at these positions.
# This grabs all the genotypes, train and test genotypes.
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

# Now we want to fill a matrix of X.
# We'll build it by first building the single effects, and then we'll add on the epistatic effects.
# There are N + M positions, where N is the number of single effects, and M is the number of epistatic effects.

# First create the genotype numpy matrix.
X_train = np.ones((train_num_usable_spores,len(best_single_positions) + len(best_epistasis_positions)))
X_train_2 = np.ones((train_num_usable_spores,len(best_single_positions) + len(best_epistasis_positions)+1))
X_test = np.ones((test_num_usable_spores,len(best_single_positions) + len(best_epistasis_positions)))

# Build the single effects matrix
for pos_index in range(len(best_single_positions)):
	pos = int(best_single_positions[pos_index])
	chr_qtl = np.searchsorted(np.array(chrom_startpoints),pos+0.5)
	start_of_chr = chrom_startpoints[chr_qtl-1]
	pos_in_chr = pos - start_of_chr

	pos_line = genotypes_file[chr_qtl-1][pos_in_chr]
	
	train_line = np.take(pos_line, train_perm)
	train_line = train_line[~np.isnan(train_set)]

	test_line = np.take(pos_line,test_perm)
	test_line = test_line[~np.isnan(test_set)]

	X_train[:,pos_index] = train_line.copy()
	X_test[:,pos_index] = test_line.copy()

	X_train_2[:,pos_index] = train_line * np.sqrt(train_errors)

# Add on the epistatic genotype matrix
for pos_pairs in range(len(best_epistasis_positions)):
	epistatic_term = best_epistasis_positions[pos_pairs][1:] # Remove the brackets [ A,B ]
	epistatic_term = epistatic_term[:-1]

	# Now split into the two positions
	epistatic_term_arr = epistatic_term.split(",")
	pos_1 = int(epistatic_term_arr[0])
	pos_2 = int(epistatic_term_arr[1])

	chr_qtl_1 = np.searchsorted(np.array(chrom_startpoints), pos_1+0.5)
	chr_qtl_2 = np.searchsorted(np.array(chrom_startpoints), pos_2+0.5)
	start_of_chr_1 = chrom_startpoints[chr_qtl_1-1]
	start_of_chr_2 = chrom_startpoints[chr_qtl_2-1]

	pos_in_chr_1 = pos_1 - start_of_chr_1
	pos_in_chr_2 = pos_2 - start_of_chr_2

	pos_line_1 = genotypes_file[chr_qtl_1-1][pos_in_chr_1]
	pos_line_2 = genotypes_file[chr_qtl_2-1][pos_in_chr_2]

	pos_line_1_train = np.take(pos_line_1,train_perm)
	pos_line_2_train = np.take(pos_line_2,train_perm)
				
	pos_line_1_train = pos_line_1_train[~np.isnan(train_set)]
	pos_line_2_train = pos_line_2_train[~np.isnan(train_set)]
				
	epis_pos_line_train = pos_line_1_train * pos_line_2_train

	pos_line_1_test = np.take(pos_line_1,test_perm)
	pos_line_2_test = np.take(pos_line_2,test_perm)
				
	pos_line_1_test = pos_line_1_test[~np.isnan(test_set)]
	pos_line_2_test = pos_line_2_test[~np.isnan(test_set)]
				
	epis_pos_line_test = pos_line_1_test * pos_line_2_test

	X_test[:,pos_pairs + len(best_single_positions)] = epis_pos_line_test.copy()
	X_train[:,pos_pairs + len(best_single_positions)] = epis_pos_line_train.copy()

	X_train_2[:,pos_pairs + len(best_single_positions)] = epis_pos_line_train * np.sqrt(train_errors)

# Ok, done parsing the positions.

# The last position in the x_train_2 matrix is the intercept.
X_train_2[:,len(best_single_positions) + len(best_epistasis_positions)] = X_train_2[:,len(best_single_positions) + len(best_epistasis_positions)] * np.sqrt(train_errors)

q,r = np.linalg.qr(X_train_2)
beta = linalg.solve_triangular(r,np.dot(np.transpose(q), train_phenotypes * np.sqrt(train_errors)), check_finite=False)
initial_beta_2 = beta[0:len(best_single_positions) + len(best_epistasis_positions)]

# Ok, now we'll only look at merging the epistatic terms.
	
corr_matrix = np.corrcoef(X_train[:,len(best_single_positions):],rowvar=False)

if(len(best_epistasis_positions)==1):
	corr_matrix = np.reshape(corr_matrix,(1,1))

corr_matrix_tril = np.tril(corr_matrix,k=-1)

alphas = np.linspace(0.60,1,800)
#alphas = np.linspace(0.95,1,100)


for alpha in alphas:

	# Obtain the all by all correlation for the positions, found within the training set.
	
	corr_groups = np.argwhere(corr_matrix_tril >= alpha).tolist()

	for i in range(len(corr_groups)):
		j = i+1
		while j < len(corr_groups):
			if(np.in1d(corr_groups[i],corr_groups[j]).any()):
				merge = np.unique(np.concatenate((corr_groups[i],corr_groups[j]),0))
				#np.reshape(corr_groups[i],merge.shape)
				corr_groups[i] = merge
				
				# Delete j
				corr_groups = np.delete(corr_groups,j)
				# Restart
				j = i
			j = j + 1

	# Now populate an initial beta array with the effects.
	# Make a copy of the best_effects array
	#initial_beta = best_effects.copy()
	initial_beta = initial_beta_2.copy()

	# Now go through the merges and assign value of initial_beta
	for i in range(len(corr_groups)):
		sum = 0
		for j in range(len(corr_groups[i])):
			sum = sum + initial_beta_2[corr_groups[i][j] + len(best_single_positions)]


		for j in range(len(corr_groups[i])):
			initial_beta[corr_groups[i][j] + len(best_single_positions)] = sum/len(corr_groups[i])


	# get R2


	predicted_phenotypes = np.dot(X_test, initial_beta)
	RSS = np.sum((test_phenotypes - np.mean(test_phenotypes) - predicted_phenotypes + np.mean(predicted_phenotypes))**2)
	R2 = 1-RSS/TSS
	# Get initial R2.
	print(str(alpha) + "	" + str(R2) + "	" + str(RSS) + "	" + str(TSS))

exit()

