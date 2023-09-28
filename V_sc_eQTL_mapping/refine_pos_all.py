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

# Systematically check every positions

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
required.add_argument('-qtl_pos', help='Plain text file containing the positions of QTLs')
optional.add_argument('--oCV', help='Outside cross-validation value (k = 0-9)', type=int, default=0)
optional.add_argument('--iCV', help='Inside cross-validation value (l = 0-8)', type=int, default=0)
optional.add_argument('--model', help='Whether to fit on the training set (m = 0), on the train+test set (m = 1) or on the complete data (m = 2)', type=int, default=0)
optional.add_argument('--dir', help='Directory where intermediate files are found.', default=cwd)
#optional.add_argument('--scratch', help='Local scratch directory', default='/n/holystore01/LABS/desai_lab/Users/nnguyenba/BBQ/all_data/genomes/')
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

# First let's take care of the outside CV
sporelist = np.array(range(len(fitnesses)))
if(args.sporelist):
    sporelist = np.loadtxt(args.sporelist, dtype=int)

# First let's take care of the outside CV

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
num_usable_spores = len(phenotypes)

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

# Read in the file with positions
prev_pos = []
prev_genotypes = []
prev_pos = np.array(prev_pos)
prev_genotypes = []
if(args.qtl_pos):
    # If we have already found a QTL, then add that QTL to the one we are searching next.
    prev_pos = np.loadtxt(args.dir + "/" + args.qtl_pos, dtype=int,ndmin=1,max_rows=1)
    size_of_prev_genome = (prev_pos.size)
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

# Ok, we've now reloaded all the genotype value for the current positions
# Set up computation settings
poolcount = 32
num_chrom_to_scan = len(genotypes_file)

def refine_positions(num):
    lowest_RSS = np.Infinity
    genome_line_lowest_RSS = []
    pos_index_at_lowest_RSS = -1
    last_q = []
    start_index = chrom_startpoints[chr_of_snp-1]
    #for scan_pos in range(left_bracket,right_bracket+1):
    #    if(scan_pos % poolcount == num):
    for scan_pos in range(left_bracket + num, right_bracket + 1, poolcount):
        genome_line = genotypes_file[chr_of_snp-1][scan_pos-start_index]

        # Remove the genomes that are not in the train set
        pos_line = np.take(genome_line,train_perm)
            
        # Remove the genomes that have no phenotypes
        pos_line = pos_line[~np.isnan(train_set)]
        pos_line = np.reshape(pos_line,(num_usable_spores,1)) # A N row by 1 column matrix

        WX = pos_line * np.sqrt(np.reshape(errors,(num_usable_spores,1)))
        QtX = np.dot(np.transpose(q_down),WX) # Gets the scale for each vectors in Q.
        QtX_Q = np.einsum('ij,j->i',q_down,np.ravel(QtX))
        orthogonalized = WX-np.reshape(QtX_Q,(num_usable_spores,1)) # Orthogonalize
        new_q = orthogonalized/np.linalg.norm(orthogonalized) # Orthonormalize
        # This gets the last column of Q.
        # Now let's get the last column of R.
        # Assemble q_up
        q_upTy = np.einsum('i,i', np.ravel(new_q), phenotypes * np.sqrt(errors))
        q_upq_upTy = np.ravel(new_q) * q_upTy
        predicted_fitnesses = initial_predicted_fitnesses + q_upq_upTy/np.sqrt(errors)
            
        # Scale the intercept term
        mean_predicted_fitnesses = np.mean(predicted_fitnesses)
    
        # RSS
        RSS = np.sum((phenotypes - mean_phenotypes - predicted_fitnesses + mean_predicted_fitnesses)**2) # This is the RSS for 1:1 line.
        #print(str(scan_pos) + "    " + str(RSS))
        if(RSS < lowest_RSS):
            lowest_RSS = RSS
            genome_line_lowest_RSS = pos_line.copy()
            pos_index_at_lowest_RSS = scan_pos
            last_q = np.ravel(new_q)


    # Return the values
    return lowest_RSS,genome_line_lowest_RSS,pos_index_at_lowest_RSS, last_q

# Put the intercept
base_genotypes = np.ones((num_usable_spores,1+size_of_prev_genome))
base_genotypes[:,1:] = prev_genotypes # First index is the intercept.

q,r = np.linalg.qr(base_genotypes * np.sqrt(np.reshape(errors,(num_usable_spores,1))))

# Consider refining only in some cases to accelerate the QTL finding process.
# Strategies: Maybe only try refining positions on the same chromosome as the last identified QTL. I found that this doesn't produce optimal solutions at all step, but it may be sufficient to do this until the end, and then completely refine at the end.
# We need to sort all the positions and the genotyping array.
start_refine = time.time()


sorted_indexes = np.argsort(prev_pos) 
# prev_pos is a list of QTL positions. sorted_index gives the indexes that would sort prev_pos. So prev_pos has QTL positions (0 to 45000), such as 2,8,5,6
# So sorted_indxes returns an array of size num_QTLs, where each value is the order you would have to take the index to sort the array. in the example, it would be 0, 2, 3, 1
sorted_prev_pos = np.take(prev_pos,sorted_indexes) # sorted_prev_pos is a sorted prev_pos array. # We then sort prev_pos with the sorted index. Returns an array 2,5,6,8
reverse_sorted_indexes = np.argsort(sorted_indexes) # This reverses the sorting procedure, which returns the 'order' of the original list, or the position where the value has ended up in. It would be: 0, 3, 1, 2

# We sorted the positions so that we know where to refine the positions of a QTL.
iterations_max = len(sorted_prev_pos)*2
lowest_RSS = np.Infinity
previous_refined = -1
for iterations in range(iterations_max):
    sorted_pos_index = np.random.randint(0, len(sorted_prev_pos)) # Numpy is range exclusive (can never return len(sorted_prev_pos) in this case).
    chr_of_snp = np.searchsorted(np.array(chrom_startpoints),sorted_prev_pos[sorted_pos_index] + 0.5) # Deal with ties

    if(sorted_pos_index == previous_refined):
        continue

    previous_refined = sorted_pos_index


    # Column downdating of QR.
    q_down,r_down = linalg.qr_delete(q,r, sorted_indexes[sorted_pos_index]+1,1,"col",check_finite=False)
    initial_beta = linalg.solve_triangular(r_down,np.dot(np.transpose(q_down), phenotypes * np.sqrt(errors)), check_finite=False) # 3.49s for 10000 loops # Beta for the WEIGHTED phenotypes.
    # first beta index is the intercept term.
    initial_predicted_fitnesses = np.dot(q_down,np.dot(r_down,initial_beta))*1/np.sqrt(errors) # Optimal multiplication order # Obtain the predicted fitnesses in the unweighted world.

    # Now we go through the bracketed positions, and obtain the likelihoods for every position
    # First, let's check the left and right side of the position of interest. If the likelihood is worse, we do not update.
    left_bracket = chrom_startpoints[chr_of_snp-1] # Minimally the beginning of the chromosome
    if (sorted_prev_pos[sorted_pos_index]-16 > left_bracket):
        left_bracket = sorted_prev_pos[sorted_pos_index]-16
    if (sorted_pos_index > 0):
        if (sorted_prev_pos[sorted_pos_index-1]+1 > left_bracket):
            left_bracket = sorted_prev_pos[sorted_pos_index-1]+1

    # Now find the right bracket
    right_bracket = chrom_endpoints[chr_of_snp-1]
    if (sorted_prev_pos[sorted_pos_index]+16 < right_bracket):
        right_bracket = sorted_prev_pos[sorted_pos_index]+16
    if (sorted_pos_index < len(sorted_prev_pos)-1):
        if (sorted_prev_pos[sorted_pos_index+1]-1 < right_bracket):
            right_bracket = sorted_prev_pos[sorted_pos_index+1]-1

    left_bracket = int(left_bracket)
    right_bracket = int(right_bracket)

    p = Pool(poolcount)
    results = p.map(refine_positions, range(poolcount))
    p.close()
    p.join()

    # Parse the results
    lowest_RSS = np.Infinity
    genome_line_lowest_RSS = []
    pos_index_at_lowest_RSS = -1
    for i in range(len(results)):
        RSS = results[i][0]
        if(RSS < lowest_RSS):
            lowest_RSS = RSS
            genome_line_lowest_RSS = results[i][1]
            pos_index_at_lowest_RSS = results[i][2]
    # We now have the best results.
    # Did the position change? If not, then we do nothing.
    if(pos_index_at_lowest_RSS != sorted_prev_pos[sorted_pos_index]):
        likelihood = num_usable_spores * math.log(lowest_RSS/num_usable_spores)
        print("Attempted to refine QTL (" + str(iterations) + "). Took : " + str(time.time() - start_refine) + " seconds. Likelihood: " + str(likelihood), file=sys.stderr)
        # Else, we update the position
        sorted_prev_pos[sorted_pos_index] = pos_index_at_lowest_RSS
        
        # We update the genome line
        prev_genotypes[:,sorted_indexes[sorted_pos_index]] = genome_line_lowest_RSS[:,0]

        # Update the QR
        q,r = linalg.qr_insert(q_down,r_down,genome_line_lowest_RSS * np.sqrt(np.reshape(errors,(num_usable_spores,1))),sorted_indexes[sorted_pos_index]+1,'col',check_finite=False) # Update the QR decomposition.


# Done iterating.
# Obtain betas

beta = linalg.solve_triangular(r,np.dot(np.transpose(q), phenotypes * np.sqrt(errors)), check_finite=False)
prev_pos = sorted_prev_pos[reverse_sorted_indexes]

print(*prev_pos)
print(*beta[1:len(beta)])


exit()

            
