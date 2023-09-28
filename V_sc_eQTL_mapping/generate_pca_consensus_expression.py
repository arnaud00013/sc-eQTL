#import libraries
import sys
import time
from multiprocessing import Pool, TimeoutError, Array
from contextlib import closing
import csv
import gzip
import os
import scipy.io
from scipy.stats import mannwhitneyu, linregress, rankdata, skew
#from scipy.stats import boxcox
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from random import randint, seed
from collections import Counter
from math import factorial, log, exp
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
import scprep

import multiprocessing as mp
from multiprocessing import Pool

#import main workspace absolute path
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[1]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[2]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[3]
nb_partitions = 300
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

# expression data
		
#matrix of gene expression (import + normalize + only select genes with finite normalized values and non-null standard deviation across cells)
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
backup_mtx_gene_expression = mtx_gene_expression
selected_genes_in_expr_mat1 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]).tolist(),np.sum(mtx_gene_expression==0,axis=0).tolist()[0]) if j!=np.shape(mtx_gene_expression)[0] ]
mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0)
selected_genes_in_expr_mat2 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]),((np.isfinite(mtx_gene_expression)).sum(axis=0).tolist())[0]) if j==np.shape(mtx_gene_expression)[0] ]
selected_genes_in_expr_mat = [int(x) for x in np.intersect1d(selected_genes_in_expr_mat1,selected_genes_in_expr_mat2)]
mtx_gene_expression = backup_mtx_gene_expression[:,selected_genes_in_expr_mat]
tmp_mtx_gene_expr = mtx_gene_expression

features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
selected_gene_names = np.array(gene_names)[selected_genes_in_expr_mat].tolist()

#keep genes that are not noisy
#import mtx_expression_PCs_weighted_R2
df_expression_PCs_weighted_R2 = pd.read_csv("{0}/mtx_weighted_Expression_PCs_R2_corr_with_HMM_genotypes_0.csv".format(cellranger_outs_folder),sep="\t",header=None)
for the_ind_partition in np.arange(1,nb_partitions,1):
    df_expression_PCs_weighted_R2 = pd.concat([df_expression_PCs_weighted_R2,pd.read_csv("{0}/mtx_weighted_Expression_PCs_R2_corr_with_HMM_genotypes_{1}.csv".format(cellranger_outs_folder,the_ind_partition),sep="\t",header=None,index_col=0)],axis=0)
mtx_expression_PCs_weighted_R2 = df_expression_PCs_weighted_R2.to_numpy()
print("np.shape(mtx_expression_PCs_weighted_R2):")
print(np.shape(mtx_expression_PCs_weighted_R2))

    #pca expression only to exclude noisy genes
pca_loadings_expression_only_ALL_pcs = PCA(n_components=np.shape(tmp_mtx_gene_expr)[1]).fit_transform(X=tmp_mtx_gene_expr)
print("np.shape(pca_loadings_expression_only_ALL_pcs):")
print(np.shape(pca_loadings_expression_only_ALL_pcs))
fit_pca_expression_only_ALL_pcs = PCA(n_components=np.shape(tmp_mtx_gene_expr)[1])
fit_pca_expression_only_ALL_pcs.fit(X=tmp_mtx_gene_expr)
lst_explained_variance_ALL_pcs_expression_only = fit_pca_expression_only_ALL_pcs.explained_variance_ratio_
mtx_genes_pca_loadings = fit_pca_expression_only_ALL_pcs.components_
print("len(lst_explained_variance_ALL_pcs_expression_only):")
print(len(lst_explained_variance_ALL_pcs_expression_only))
'''
#find the number of PCs explaining 99% of the variance
cumul_var_expl = 0
nb_expression_PCs = 1
while cumul_var_expl < 0.99:
    cumul_var_expl = cumul_var_expl + (lst_explained_variance_ALL_pcs_expression_only[nb_expression_PCs-1])
    nb_expression_PCs = nb_expression_PCs + 1
print("Number of expression PCs saved = {0}".format(nb_expression_PCs))
'''
nb_expression_PCs = np.shape(mtx_expression_PCs_weighted_R2)[0]
mtx_genes_pca_loadings = mtx_genes_pca_loadings[np.arange(nb_expression_PCs).tolist(),:]

    #Compute genes signal scores and loadings skewness
arr_genes_signal_scores = np.zeros(np.shape(mtx_genes_pca_loadings)[1])
arr_genes_loading_skewness = np.zeros(np.shape(mtx_genes_pca_loadings)[1])
for ind_gene in np.arange(np.shape(mtx_genes_pca_loadings)[1]):
    arr_genes_signal_scores[ind_gene] = np.sum(mtx_expression_PCs_weighted_R2[:,2]*np.abs(mtx_genes_pca_loadings[:,ind_gene]))
    arr_genes_loading_skewness[ind_gene] = abs(skew(mtx_genes_pca_loadings[:,ind_gene]))
    #Keep genes that in the 75% superior part of the signal score distribution
threshold_signal_score = np.quantile(arr_genes_signal_scores, 0.25)
selected_genes_in_expr_mat3 = [i for i in np.arange(len(arr_genes_signal_scores)) if arr_genes_signal_scores[i] > threshold_signal_score]
selected_genes_in_expr_mat = [int(x) for x in np.intersect1d(selected_genes_in_expr_mat,selected_genes_in_expr_mat3)]
mtx_gene_expression = mtx_gene_expression[:,selected_genes_in_expr_mat]
del tmp_mtx_gene_expr

pd.DataFrame(mtx_gene_expression).to_csv("{0}/Normalized_mtx_gene_expression.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)

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

pd.DataFrame(mean_E).to_csv("{0}/Consensus_mtx_gene_expression.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)

#Perform pca on consensus expression
pca_consensus_expression = PCA(n_components=np.shape(mean_E)[1]).fit_transform(X=mean_E)
pd.DataFrame(pca_consensus_expression).to_csv("{0}/PCA_consensus_expression.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)