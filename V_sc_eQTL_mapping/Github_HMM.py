#import libraries
import string
import numpy as np
import sys
import csv
import itertools
import time
import scipy.stats as st
from multiprocessing import Pool, TimeoutError
from contextlib import closing
import os
import pandas as pd
from random import sample, seed
from math import factorial, log, exp
import matplotlib.pyplot as plt

#import main workspace absolute path
workspace_path = sys.argv[1]
yeast_project_wp_path = sys.argv[2]
nb_cpus = int(sys.argv[3]) #16
max_ID_subset = int(sys.argv[4])
sys.path.append(workspace_path)
from spore_defs import *


def haldane_chrom(d,chrom):
    # distance d is in basepairs
    # 1 cM = 2 kb
    #rates_by_chrom_old = [1384.04365247,2436.76980165,1693.43573485,2276.48993531,2311.91848742,1457.01815933,2471.6244856,2131.20772674,
                                #2128.15505231,2376.75200613,2291.77716017,2438.58106786,2539.99772823,2352.42233483,2223.93613249,2297.38772424]
                                
    rates_by_chrom = [1578.14349582,
                            2508.96380702,
                            1877.25014075,
                            2312.4896111,
                            2326.29062162,
                            1939.5698633,
                            2503.27060554,
                            2173.41070953,
                            2225.02620068,
                            2389.99027447,
                            2315.98587503,
                            2470.58724189,
                            2502.60184574,
                            2467.5457676,
                            2551.69201858,
                            2412.52317237]
    
    return 0.5*(1-np.exp(-2*d*(0.01/rates_by_chrom[chrom])))

def haldane(d):
    # distance d is in basepairs
    # 1 cM = 2 kb
    rate = 2000.0
    
    return 0.5*(1-np.exp(-2*d*(0.01/rate)))

        
# def inv_haldane(p,d):
#     return -1*np.log(1-2*p)/(0.01*d/2000)
#
# def inv_haldane_approx(SNP_map,p,chrom,i):
#     l = SNP_map.get_SNP_dist_between(chrom,i+1,i)
#     return l/(p*100.0)
    

def get_transition_matrices(SNP_list,chrom):
    length = len(SNP_list[chrom])-1
    T = np.full((length,4,4),np.nan)

    error_prob = 0.0037
    return_prob = 0.3

    for i in range(length):
        dist = SNP_list[chrom][i+1]-SNP_list[chrom][i]
        pr  = haldane_chrom(dist,chrom)
        #pr  = haldane(dist)
        # Upper left: recombination probabilities between correct states
        T[i][0][0] = 1-pr-error_prob
        T[i][1][0] = pr
        T[i][0][1] = pr
        T[i][1][1] = 1-pr-error_prob
        # Upper right: prob to enter error states
        T[i][0][2] = error_prob
        T[i][0][3] = 0
        T[i][1][2] = 0
        T[i][1][3] = error_prob
        # Lower left: Prob to return from error states
        T[i][2][0] = return_prob
        T[i][2][1] = 0
        T[i][3][0] = 0
        T[i][3][1] = return_prob
        # Lower right: Prob to stay in error states
        T[i][2][2] = 1-return_prob
        T[i][2][3] = 0
        T[i][3][2] = 0
        T[i][3][3] = 1-return_prob

    return(T)
    
def get_observation_probs(RM_reads,BY_reads,total_avg_reads):
    O = np.full([len(RM_reads),4,4],0.0)

    nRM = np.nan_to_num(RM_reads)
    nBY = np.nan_to_num(BY_reads)
    N = nRM+nBY
    p_error = 0.01
    
    for i in range(len(N)):
        if total_avg_reads > 1 and N[i] > 10*int(total_avg_reads): 
            N[i] = 0
            nRM[i] = 0
            nBY[i] = 0
        
    pRM = st.binom.pmf(nRM,N,(1-p_error))
    pBY = st.binom.pmf(nRM,N,p_error)
        
    O[:,0,0] = pRM[:]
    O[:,1,1] = pBY[:]
    O[:,2,2] = pBY[:]
    O[:,3,3] = pRM[:]        
            
    return(O)

def forward(T,O):
    fprobs_norm = np.full((len(O),4),np.nan,dtype=np.longdouble)
    f_log_weights = np.full(len(fprobs_norm),np.nan,dtype=np.longdouble)
    fprobs_norm[0][0] = 0.49
    fprobs_norm[0][1] = 0.49
    fprobs_norm[0][2] = 0.01
    fprobs_norm[0][3] = 0.01
    f_log_weights[0] = np.log(sum(fprobs_norm[0]))
    for i in range(1,len(fprobs_norm)):
        fprobsi = np.dot(np.dot(fprobs_norm[i-1],T[i-1]),O[i])
        weight = sum(fprobsi)
        if weight == 0:
            print(fprobs_norm[i-1])
            print(T[i-1])
            print(O[i])
            print(i)
            #sys.exit()
        fprobs_norm[i] = fprobsi/weight
        f_log_weights[i] = np.log(weight)+f_log_weights[i-1]
    if np.any(np.isnan(fprobs_norm)):
        print(fprobsi)
        print(weight)
    return(fprobs_norm,f_log_weights)
    
def backward(T,O):
    bprobs_norm = np.full((len(O),4),np.nan)
    b_log_weights = np.full(len(bprobs_norm),np.nan)
    bprobs_norm[-1][0] = 1.0
    bprobs_norm[-1][1] = 1.0
    bprobs_norm[-1][2] = 1.0
    bprobs_norm[-1][3] = 1.0
    b_log_weights[-1] = 0
    for i in range(-2,-1*len(bprobs_norm)-1,-1):
        bprobsi = np.dot(np.dot(T[i+1],O[i+1]),bprobs_norm[i+1])
        weight = sum(bprobsi)
        bprobs_norm[i] = bprobsi/weight
        b_log_weights[i] = np.log(weight)+b_log_weights[i+1]
            
    return(bprobs_norm,b_log_weights)
    
def posteriors(fprobs_norm,f_log_weights,bprobs_norm,b_log_weights,T,O):

    total_log_L = f_log_weights[-1]

    # initialize vectors for posteriors
    log_posteriors = np.full((len(fprobs_norm),4),np.nan)
    post_norms = np.full(len(fprobs_norm),np.nan)
    
    # intialize vector for recomb map quantities
    post_RM_norms = np.full(len(fprobs_norm),np.nan)
    post_BY_norms = np.full(len(fprobs_norm),np.nan)
    post_RMerr_norms = np.full(len(fprobs_norm),np.nan)
    post_BYerr_norms = np.full(len(fprobs_norm),np.nan)
    
    trans_RM_BY = np.full(len(T),np.nan)
    trans_BY_RM = np.full(len(T),np.nan)    
            
    for s in range(4):
        log_posteriors[:,s] = np.log(fprobs_norm[:,s])+f_log_weights[:]+np.log(bprobs_norm[:,s])+b_log_weights[:]-total_log_L
                
    post_norms[:] = np.exp(log_posteriors[:,0])+np.exp(log_posteriors[:,2])
    
    post_RM_norms[:] = np.exp(log_posteriors[:,0])
    post_BY_norms[:] = np.exp(log_posteriors[:,1])
    post_RMerr_norms[:] = np.exp(log_posteriors[:,2])
    post_BYerr_norms[:] = np.exp(log_posteriors[:,3])
    
    # add to recombination counts for recomb map
    trans_RM_BY[:] = np.exp(np.log(fprobs_norm[:-1,0])+f_log_weights[:-1]+np.log(T[:,0,1])+np.log(bprobs_norm[1:,1])+b_log_weights[1:]+np.log(O[1:,1,1])-total_log_L) 
    trans_BY_RM[:] = np.exp(np.log(fprobs_norm[:-1,1])+f_log_weights[:-1]+np.log(T[:,1,0])+np.log(bprobs_norm[1:,0])+b_log_weights[1:]+np.log(O[1:,0,0])-total_log_L) 
    
    
    return(log_posteriors,post_norms,post_RM_norms,post_BY_norms,post_RMerr_norms,post_BYerr_norms,trans_RM_BY,trans_BY_RM)
    

#######    

# Program to infer missing SNPs using HMM forward-backward algorithm
# Uses SNP location map and sequenced genome
t0 = time.time()


#import data
df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"
df_best_matches = pd.read_csv("{0}/backup_expected_distance_df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })

#Dimensionality reduction clusters
barcode_best_dimred_clusters = pd.read_csv("{0}/clusters_MS_PHATE_top500pcs_genotype_only.csv".format(workspace_path),sep="\t",header=None,dtype={ '0': int })
barcode_best_dimred_clusters.columns = ["cell"]
lst_barcode_best_dimred_clusters = barcode_best_dimred_clusters["cell"].tolist()
del barcode_best_dimred_clusters
#find duplicates for the assigned clusters
lst_clusters_not_yet_duplicated = []
lst_clusters_already_duplicated = []
for current_cluster in lst_barcode_best_dimred_clusters:
    if (lst_barcode_best_dimred_clusters.count(current_cluster) > 1):
        if (not current_cluster in lst_clusters_already_duplicated):
            lst_clusters_not_yet_duplicated.append(True)
            lst_clusters_already_duplicated.append(current_cluster)
        else:
            lst_clusters_not_yet_duplicated.append(False)
    else:
        lst_clusters_not_yet_duplicated.append(True)

lst_supercells_clusters = np.array(lst_barcode_best_dimred_clusters)[lst_clusters_not_yet_duplicated].tolist()
nb_supercells_clusters = len(lst_supercells_clusters)

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

# Read SNP map
SNP_reader = csv.reader(open("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),'r'),delimiter='\t')

#genome_str = genome_str_to_int(next(SNP_reader))
SNP_list = []
for current_chromosome in lst_label_chromosomes:
    df_current_chr_snps = df_pos_snps[df_pos_snps.chromosome==current_chromosome]
    SNP_list.append(df_current_chr_snps.position.tolist())
num_chroms = len(SNP_list)
num_SNPs = [len(x) for x in SNP_list]
num_SNPs_total = sum(num_SNPs)
print(num_SNPs)
print(num_SNPs_total)
#sys.stdout.flush()
#chrom_startpoints = get_chrom_startpoints(genome_str)
#chrom_endpoints = get_chrom_endpoints(genome_str)

chrom_startpoints = [0, 996, 4732, 5291, 9327, 11187, 12476, 16408, 18047, 20126, 23101, 26341, 30652, 33598, 35398, 39688]
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41608]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1921]

#dict_BY_count = {}
#dict_RM_count = {}
dict_HMM_posterior_probs = {}

i=0
for current_chromosome in lst_label_chromosomes:
    #dict_BY_count[current_chromosome] = pd.read_csv("{0}/data/good_cells_genotypes/BY_count_{1}.csv".format(workspace_path,current_chromosome),sep="\t",header=0,index_col=0).to_numpy()
    #dict_RM_count[current_chromosome] = pd.read_csv("{0}/data/good_cells_genotypes/RM_count_{1}.csv".format(workspace_path,current_chromosome),sep="\t",header=0,index_col=0).to_numpy()

    #initialize matrix of HMM genotypes
    dict_HMM_posterior_probs[current_chromosome] = np.zeros(shape=(nb_supercells_clusters,num_SNPs[i]))
    i = i + 1

# Get transition matrices
transition_matrices = []
for i in range(num_chroms):
    transition_matrices.append(get_transition_matrices(SNP_list,i))
print('Done finding transition matrices.')
#sys.stdout.flush()

# loop over barcodes
for i in np.arange(len(lst_supercells_clusters)):
    #import counts data for current barcode
    the_cluster = lst_supercells_clusters[i]
    df_current_barcode_counts = pd.read_csv("{0}/count_clusters_supercells/df_SNP_count_supercell_cluster_{1}.csv".format(workspace_path,the_cluster),sep="\t",header=0)
    #Get coverage stats
    all_reads_RM = np.sum(df_current_barcode_counts.RM_SNP_reads_count)
    all_reads_BY = np.sum(df_current_barcode_counts.BY_SNP_reads_count)
    total_len = np.shape(df_current_barcode_counts)[0]
    total_nonzeros = np.count_nonzero(df_current_barcode_counts.RM_SNP_reads_count+df_current_barcode_counts.BY_SNP_reads_count)

    coverage = float(total_nonzeros)/float(total_len)
    total_avg_reads = (all_reads_RM+all_reads_BY)/float(total_len)
    print("Barcode {0} average depth of coverage = {1} and breadth of coverage = {2}!".format(lst_supercells_clusters[i],total_avg_reads,coverage))

    id_chrom = 0
    # Loop over chromosomes for HMM
    for current_chromosome in lst_label_chromosomes:
        df_current_barcode_counts_in_current_chr = df_current_barcode_counts[df_current_barcode_counts.chromosome==current_chromosome]
        # Pick out reads for this current_chromosome
        all_reads_RM = df_current_barcode_counts_in_current_chr.RM_SNP_reads_count
        all_reads_BY = df_current_barcode_counts_in_current_chr.BY_SNP_reads_count

        # Pick out transition matrices for this current_chromosome
        T = transition_matrices[id_chrom]
                
        # Get observation matrices
        O = get_observation_probs(all_reads_RM,all_reads_BY,total_avg_reads)
        if np.isnan(O).any():
            print('Error! nan in observation probabilites, '+str(current_chromosome)+'. ',sys.stderr)
            #sys.stderr.flush()
            print(i)     
            break    
                
        # Get forward & backward probabilities
        fprobs_norm,f_log_weights = forward(T,O)     
        bprobs_norm,b_log_weights = backward(T,O)
        if np.isnan(fprobs_norm).any():
            print('Error! nan in fprobs, '+str(current_chromosome)+' of cluster '+str(the_cluster)+'.',sys.stderr)
            #sys.stderr.flush()
            print(i)
            break    
        if np.isnan(bprobs_norm).any():
            print ('Error! nan in bprobs, '+str(current_chromosome)+'. ',sys.stderr)
            #sys.stderr.flush()
            print(i)
            break    

        # Get total log likelihood
        log_likelihood = f_log_weights[-1]

        # Get posterior probs 
        log_posteriors,post_probs,post_probs_RM,post_probs_BY,post_probs_RMerr,post_probs_BYerr,trans_chrom_RM_BY,trans_chrom_BY_RM = posteriors(fprobs_norm,f_log_weights,bprobs_norm,b_log_weights,T,O)
        
        if np.isnan(post_probs).any():
            print('Error! nan in inferred genome, '+str(current_chromosome)+'. ',sys.stderr)
            #sys.stderr.flush()    
            print(i)
            break    
        #save HMM genotype
        dict_HMM_posterior_probs[current_chromosome][i,:] = post_probs
        id_chrom = id_chrom + 1
for current_chromosome in lst_label_chromosomes:
    #save matrix of genotypes for current barcode
    pd.DataFrame(dict_HMM_posterior_probs[current_chromosome]).to_csv("{0}/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(data_wp_path,current_chromosome), sep='\t',na_rep="NA",header=False,index=False)
    
tf = time.time()

print("t0:")
print(t0)
print("tf:")
print(tf)
