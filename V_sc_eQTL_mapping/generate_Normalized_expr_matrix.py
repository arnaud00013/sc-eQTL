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
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[3]
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
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41593]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1906]
#exit()

# expression data
'''
#matrix of gene expression (import + normalize + only select genes with finite normalized values, non-null standard deviation across cells and top75% signal score)
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0)

#keep genes that are not noisy
    #import expressed genes dataframe
df_expressed_genes_selected = pd.read_csv("{0}/df_expressed_genes_selected.csv".format(workspace_path), sep='\t',header=0)
df_expressed_genes_selected['sc_eQTL_gene_id'] = np.arange(np.shape(df_expressed_genes_selected)[0]).tolist()
#pd.DataFrame(df_expressed_genes_selected).to_csv("{0}/df_expressed_genes_selected.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)
selected_genes_in_expr_mat = df_expressed_genes_selected['Expressed_gene_original_index'].tolist()
mtx_gene_expression = mtx_gene_expression[:,selected_genes_in_expr_mat]
'''

#matrix of gene expression (import + normalize + only select genes with finite normalized values and non-null standard deviation across cells)
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
mtx_gene_expression = np.squeeze(np.asarray(mtx_gene_expression))
mtx_gene_expression = mtx_gene_expression/np.median(np.sum(mtx_gene_expression,axis=1))
mtx_gene_expression = np.log1p(mtx_gene_expression)
mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0)
selected_genes_in_expr_mat1 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]).tolist(),np.sum(mtx_gene_expression==0,axis=0).tolist()) if j!=np.shape(mtx_gene_expression)[0] ]
selected_genes_in_expr_mat2 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]),((np.isfinite(mtx_gene_expression)).sum(axis=0).tolist())) if j==np.shape(mtx_gene_expression)[0] ]
selected_genes_in_expr_mat = [int(x) for x in np.intersect1d(selected_genes_in_expr_mat1,selected_genes_in_expr_mat2)]
mtx_gene_expression = mtx_gene_expression[:,selected_genes_in_expr_mat]
features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
selected_gene_names = np.array(gene_names)[selected_genes_in_expr_mat].tolist()
#save metadata about selected expressed genes names
data = {'Expressed_gene_original_index' : selected_genes_in_expr_mat,
        'sc_eQTL_gene_id' : np.arange(len(selected_genes_in_expr_mat)).tolist(),
        'Gene_name':selected_gene_names }

    #pca expression only to exclude noisy genes
pca_loadings_expression_only_ALL_pcs = PCA(n_components=np.shape(mtx_gene_expression)[1]).fit_transform(X=mtx_gene_expression)
print("np.shape(pca_loadings_expression_only_ALL_pcs):")
print(np.shape(pca_loadings_expression_only_ALL_pcs))
fit_pca_expression_only_ALL_pcs = PCA(n_components=np.shape(mtx_gene_expression)[1])
fit_pca_expression_only_ALL_pcs.fit(X=mtx_gene_expression)
lst_explained_variance_ALL_pcs_expression_only = fit_pca_expression_only_ALL_pcs.explained_variance_ratio_
mtx_genes_pca_loadings = fit_pca_expression_only_ALL_pcs.components_
print("len(lst_explained_variance_ALL_pcs_expression_only):")
print(len(lst_explained_variance_ALL_pcs_expression_only))
print("lst_explained_variance_ALL_pcs_expression_only[0:10]:")
print(lst_explained_variance_ALL_pcs_expression_only[0:10])
#find the number of PCs explaining 99% of the variance
cumul_var_expl = 0
nb_expression_PCs = 1
while cumul_var_expl < 0.99:
    cumul_var_expl = cumul_var_expl + (lst_explained_variance_ALL_pcs_expression_only[nb_expression_PCs-1])
    nb_expression_PCs = nb_expression_PCs + 1
print("Number of expression PCs saved = {0}".format(nb_expression_PCs))

#SAVE OUTPUT FILES
pd.DataFrame(mtx_gene_expression).to_csv("{0}/Normalized_mtx_gene_expression.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)
pd.DataFrame(data).to_csv("{0}/df_expressed_genes_selected.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)
pd.DataFrame(pca_loadings_expression_only_ALL_pcs).to_csv("{0}/pca_expression_only_ALL_pcs.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)
