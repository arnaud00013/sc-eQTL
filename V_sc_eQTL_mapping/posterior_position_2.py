# Trying faster code for this.

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

required.add_argument('--fit', help='Phenotype data / Expression matrix in the case of sc_eQTL')
required.add_argument('--geneid', help='Gene id',type=int)
optional.add_argument('-qtl_pos', help='Plain text file containing the positions of QTLs', required=True)
optional.add_argument('--oCV', help='Outside cross-validation value (k = 0-9)', type=int, default=0)
optional.add_argument('--iCV', help='Inside cross-validation value (l = 0-8)', type=int, default=0)
optional.add_argument('--model', help='Whether to fit on the training set (m = 0), on the train+test set (m = 1) or on the complete data (m = 2)', type=int, default=0)
optional.add_argument('--dir', help='Directory where qtl_pos and qtl_geno are found', default=cwd)
optional.add_argument('--scratch', help='Local scratch directory', default='/n/holystore01/LABS/desai_lab/Users/nnguyenba/BBQ/all_data/genomes/')
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

if(len(fitnesses_data.shape) != 2 or args.unweighted == 1):
    # No errors found, assume all errors the same.
    if(len(fitnesses_data.shape) == 1):
        fitnesses_data = np.reshape(fitnesses_data,(-1,1))

    # No errors found, assume all errors the same.
    fitnesses = fitnesses_data[:,0]
    errors = np.ones(len(fitnesses_data))
else:
    fitnesses = fitnesses_data[:,0]
    errors = fitnesses_data[:,1]

errors = np.square(errors)
errors = np.reciprocal(errors)

seed = 100000
np.random.seed(seed) # Train set 1

sporelist = np.array(range(len(fitnesses)))
if(args.sporelist):
    sporelist = np.loadtxt(args.sporelist, dtype=int)

if(args.downsample > 0 and args.downsample < len(sporelist)):
    #fitnesses = fitnesses[0:args.downsample]
    #errors = errors[0:args.downsample]
    sporelist = sporelist[0:args.downsample]

perm = np.random.permutation(sporelist)


train_perm = perm.copy()

if(args.model != 2):
    train_perm = np.delete(train_perm, np.r_[outside_CV/10 * len(sporelist):(outside_CV + 1)/10 * len(sporelist)].astype(int),axis=0)
    validation_perm = np.take(perm, np.r_[outside_CV/10 * len(sporelist):(outside_CV + 1)/10 * len(sporelist)].astype(int))

    if(args.model != 1):
        # Ok now let's take care of the inside CV
        # To do this, we split the train_perm into a train/test permutation
        test_perm = np.take(train_perm, np.r_[inside_CV/9 * len(train_perm):(inside_CV + 1)/9 * len(train_perm)].astype(int))
        train_perm = np.delete(train_perm, np.r_[inside_CV/9 * len(train_perm):(inside_CV + 1)/9 * len(train_perm)].astype(int))
# We're doing a k*l fold validation procedure, where l = k-1.
# This allows us to only create 10 test sets, and only 10 validation sets, so the cross validation loops do not explode.
# For example, let the 80 - 10 - 10 (train - test - validation) split
# We can use the same validation for the following split: 10 - 80 -10 (test - train - validation)
# Now looking at that split, we can use the same test to do the following: 10 - 10 - 80 (test - validation - train)


# We will only 'train' on a subset of the data
train_set = np.take(fitnesses,train_perm) # This is 80% of the fitness data
errors = np.take(errors,train_perm)

phenotypes = train_set[~np.isnan(train_set)] # Is a numpy.ndarray
mean_phenotypes = np.mean(phenotypes)
TSS = np.sum((phenotypes-mean_phenotypes)**2)
errors = errors[~np.isnan(train_set)]

weighted_mean_phenotypes = np.sum(errors*phenotypes)/np.sum(errors)
W_TSS = np.sum(errors*(phenotypes-weighted_mean_phenotypes)**2)

num_usable_spores = len(phenotypes)


# Read in the file with positions
prev_pos = []
prev_genotypes = []
prev_pos = np.array(prev_pos)
prev_genotypes = []
if(args.qtl_pos):
    # If we have already found a QTL, then add that QTL to the one we are searching next.
    prev_pos = np.loadtxt(args.dir + "/" + args.qtl_pos, dtype=int,ndmin=1,max_rows=1)
    effects = np.loadtxt(args.dir + "/" + args.qtl_pos, dtype=float,ndmin=1,skiprows=1)
    
    prev_genotypes = np.zeros((num_usable_spores,len(prev_pos)))

    # Have the positions, now assembly a matrix of genotypes    
    for pos_index in range(len(prev_pos)):
        pos = prev_pos[pos_index]
        chr_qtl = np.searchsorted(np.array(chrom_startpoints), pos + 0.5)
        pos_in_chr = pos - chrom_startpoints[chr_qtl-1]

        pos_line = genotypes_file[chr_qtl-1][pos_in_chr]
        pos_line = np.take(pos_line,train_perm)
        pos_line = pos_line[~np.isnan(train_set)]

        prev_genotypes[:,pos_index] = pos_line.copy()
        

else:
    exit()



# Now let's work with the positions and the genotypes.
# First, let's get an array of the sorted positions.
sorted_indexes = np.argsort(prev_pos)
sorted_prev_pos = np.take(prev_pos,sorted_indexes)
sorted_effects = np.take(effects,sorted_indexes)
sorted_genotypes = np.take(prev_genotypes,sorted_indexes,axis=1)

#print(sorted_prev_pos)

poolcount = 32

def get_likelihood(num):
    likelihoods = []
    start_index = chrom_startpoints[chr_of_snp-1]
    for scan_pos in range(left_bracket,right_bracket+1):
        if(scan_pos % poolcount == num):
            pos_line = genotypes_file[chr_of_snp-1][scan_pos-start_index]

            # Remove the genomes that are not in the train set
            pos_line = np.take(pos_line,train_perm)
            
            # Remove the genomes that have no phenotypes
            pos_line = pos_line[~np.isnan(train_set)]
            pos_line = np.reshape(pos_line,(num_usable_spores,1)) # A N row by 1 column matrix

            WX = pos_line * np.sqrt(np.reshape(errors,(num_usable_spores,1)))
            QtX = np.dot(np.transpose(q),WX) # Gets the scale for each vectors in Q.
            QtX_Q = np.einsum('ij,j->i',q,np.ravel(QtX))
            orthogonalized = WX-np.reshape(QtX_Q,(num_usable_spores,1)) # Orthogonalize
            new_q = orthogonalized/np.linalg.norm(orthogonalized) # Orthonormalize
            # This gets the last column of Q.
            # We only need the last column of Q to get the new residuals. We'll assemble the full Q or the full R if we need it (i.e. to obtain betas).

            q_upTy = np.einsum('i,i', np.ravel(new_q), phenotypes * np.sqrt(errors))
            q_upq_upTy = np.ravel(new_q) * q_upTy
            predicted_fitnesses = initial_predicted_fitnesses + q_upq_upTy/np.sqrt(errors)

            # Scale the intercept term
            mean_predicted_fitnesses = np.mean(predicted_fitnesses)
    
            # RSS
            RSS = np.sum((phenotypes - mean_phenotypes - predicted_fitnesses + mean_predicted_fitnesses)**2) # This is the RSS for 1:1 line.
            log_likelihood = -num_usable_spores/2 * np.log(2*math.pi) -num_usable_spores/2 * np.log(RSS/num_usable_spores) - num_usable_spores/2
            likelihoods.append(log_likelihood)


    return likelihoods


# Now, for each value in sorted_prev_pos, we'll scan positions
for pos in range(len(sorted_prev_pos)):
    # Identify the chromosome
    chr_of_snp = np.searchsorted(np.array(chrom_startpoints),sorted_prev_pos[pos]+0.5)

    # Now we go through the bracketed positions, and obtain the likelihoods for every position
    # First, let's check the left and right side of the position of interest. If the likelihood is worse, we do not update.
    left_bracket = chrom_startpoints[chr_of_snp-1] # Minimally the beginning of the chromosome
    if (sorted_prev_pos[pos]-16 > left_bracket):
        left_bracket = sorted_prev_pos[pos]-16
    if (pos > 0):
        if (sorted_prev_pos[pos-1]+1 > left_bracket):
            left_bracket = sorted_prev_pos[pos-1]+1

    # Now find the right bracket
    right_bracket = chrom_endpoints[chr_of_snp-1]
    if (sorted_prev_pos[pos]+16 < right_bracket):
        right_bracket = sorted_prev_pos[pos]+16
    if (pos < len(sorted_prev_pos)-1):
        if (sorted_prev_pos[pos+1]-1 < right_bracket):
            right_bracket = sorted_prev_pos[pos+1]-1

    left_bracket = int(left_bracket)
    right_bracket = int(right_bracket)

    #print(str(left_bracket) + "    " + str(sorted_prev_pos[pos]) + "    " + str(right_bracket))


    # Ok, we now have brackets, so let's start computing likelihoods.
    # First, we need to grab the genotypes at the positions we are considering.
    # The genotype values are in sorted_genotypes.
    # So we make a copy first.
    genomes = sorted_genotypes

    # Now we remove the row that corresponds to the position we are considering
    genomes = np.delete(genomes,pos,axis=1)
    base_genotypes = np.ones((num_usable_spores,len(sorted_prev_pos)))
    base_genotypes[:,1:] = genomes

    # Generate the QR decomposition for the weighted genomes file
    q,r = np.linalg.qr(base_genotypes * np.sqrt(np.reshape(errors,(num_usable_spores,1))))

    initial_beta = linalg.solve_triangular(r,np.dot(np.transpose(q), phenotypes * np.sqrt(errors)), check_finite=False) # 3.49s for 10000 loops # Beta for the WEIGHTED phenotypes.
    # first beta index is the intercept term.
    initial_predicted_fitnesses = np.dot(q,np.dot(r,initial_beta))*1/np.sqrt(errors) # Optimal multiplication order # Obtain the predicted fitnesses in the unweighted world.

    # Now we go through the bracketed positions, and obtain the likelihoods for every position
    likelihood_array = np.zeros(right_bracket-left_bracket+1)

    p = Pool(poolcount)
    results = p.map(get_likelihood, range(poolcount))
    p.close()
    p.join()

    lowest_RSS = np.Infinity
    last_q = []

    # Ok assemble the results.
    for i in range(len(results)):
        for scan_pos in range(left_bracket,right_bracket+1):
            if(scan_pos % poolcount == i):
                likelihood_array[scan_pos-left_bracket] = results[i].pop(0)



    # Get sum of exp
    normalization = np.logaddexp.reduce(likelihood_array)
    posterior_array = np.zeros(right_bracket-left_bracket+1)
    lead_snp_posterior = 0
    lead_snp_position = -1
    for scan_pos in range(left_bracket,right_bracket+1):
        #print(str(scan_pos) + "    " + str(np.exp(likelihood_array[scan_pos-left_bracket] - normalization)))
        post = np.exp(likelihood_array[scan_pos-left_bracket] - normalization)
        posterior_array[scan_pos - left_bracket] = post

        if(post > lead_snp_posterior):
            lead_snp_posterior = post
            lead_snp_position = scan_pos
    
    # Now let's get the 95th confidence interval from the posterior probabilities.
    if(lead_snp_posterior > 0.95):
        # Lead SNP is already at 95% CI
        print(str(sorted_prev_pos[pos]) + "    " + str(sorted_prev_pos[pos]) + "    " + str(sorted_prev_pos[pos]) + "    " + str(lead_snp_posterior) + "    " + str(sorted_effects[pos]) + str(args.geneid))
        #print(str(sorted_prev_pos[pos]) + "    " + "[" + str(sorted_prev_pos[pos]) + "," + str(sorted_prev_pos[pos]) + "]" + "    " + str(lead_snp_posterior) + "    " + str(sorted_effects[pos]))
    else:
        # Ok find the left edge
        factor = lead_snp_posterior * 0.05
        left_CI_position = left_bracket
        sum_left = 0
        for position in range(left_bracket, lead_snp_position):
            sum_left = sum_left + posterior_array[position-left_bracket]

        left_CI_posterior = 0
        flag_break_left = 0
        for position in reversed(range(left_bracket, lead_snp_position)):
            left_CI_posterior = left_CI_posterior + posterior_array[position-left_bracket]
            left_CI_position = position
            if(left_CI_posterior > (0.95-factor) * sum_left):
                flag_break_left = 1
                break

        if(flag_break_left == 0):
            left_CI_position = lead_snp_position

        # Now find the right edge.
        right_CI_position = right_bracket
        sum_right = 0
        for position in range(lead_snp_position + 1, right_bracket+1):
            sum_right = sum_right + posterior_array[position-left_bracket]

        right_CI_posterior = 0
        flag_break_right = 0
        for position in range(lead_snp_position + 1, right_bracket+1):
            right_CI_posterior = right_CI_posterior + posterior_array[position-left_bracket]
            right_CI_position = position
            if(right_CI_posterior > (0.95-factor) * sum_right):
                flag_break_right = 1
                break

        if(flag_break_right == 0):
            right_CI_position = lead_snp_position

        if(sorted_prev_pos[pos] < left_CI_position):
            left_CI_position = sorted_prev_pos[pos]
        if(sorted_prev_pos[pos] > right_CI_position):
            right_CI_position = sorted_prev_pos[pos]
        print(str(sorted_prev_pos[pos]) + "    " + str(left_CI_position) + "    " + str(right_CI_position) + "    " + str(lead_snp_posterior) + "    " + str(sorted_effects[pos]) + "    " + str(args.geneid))
    #exit()


    #print(chr_of_snp)
exit()
