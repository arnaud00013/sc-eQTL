import string
import numpy as np
from scipy import linalg
import scipy.io
import sys
import csv
import itertools
import time
import random
import argparse
import os
import os.path
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
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[3]

#import spore_defs from defined workspace
sys.path.append(workspace_path)
from spore_defs import *
from argparse import ArgumentParser, SUPPRESS

#import data
df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_SNP_metadata = pd.read_csv("{0}/SNP_metadata.txt".format(cellranger_outs_folder),sep="\t",header=None,dtype={ '0': int, '1': str, '2': int, '3': str, '4': int, '5': int })
df_SNP_metadata.columns = ["raw_SNP", "chromosome","position","Allele","met1","met2"]
df_SNP_metadata["SNP_id"] = np.arange(np.shape(df_SNP_metadata)[0])
arr_raw_SNP = np.array(df_SNP_metadata['raw_SNP'].to_numpy())
arr_SNP_id_in_genome_mtx = np.array(df_SNP_metadata['SNP_id'].to_numpy())
#convert the 2 last arrays to a dictionary
dict_raw_SNP_to_id_in_genome_mtx = dict(zip(arr_raw_SNP.tolist(), arr_SNP_id_in_genome_mtx.tolist()))
df_lst_cells = pd.read_csv("{0}/lst_barcodes_with_expression_data.txt".format(workspace_path),sep="\t",header=None,dtype={ '0': str })
#import list of cells and chromosome info
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"
dic_chr_length = {"chr01":230218,"chr02":813184,"chr03":316620,"chr04":1531933,"chr05":576874,"chr06":270161,"chr07":1090940,"chr08":562643,"chr09":439888,"chr10":745751,"chr11":666816,"chr12":1078177,"chr13":924431,"chr14":784333,"chr15":1091291,"chr16":948066}
chrom_startpoints = [0, 996, 4732, 5291, 9327, 11187, 12476, 16408, 18047, 20126, 23101, 26341, 30652, 33598, 35398, 39688]
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41608]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1921]

#Save median expression of genes
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
# backup_mtx_gene_expression
    #median gene expression
lst_median_gene_expression = (np.median(mtx_gene_expression,axis=0)).tolist()
    #mean normalized gene expression
lst_mean_raw_gene_expression = (np.mean(mtx_gene_expression,axis=0)).tolist()
#Normalize gene expression
mtx_gene_expression = pd.read_csv(workspace_path+"/Normalized_mtx_gene_expression.csv",sep="\t",header=None).to_numpy()
    #mean normalized gene expression
lst_mean_norm_gene_expression = (np.mean(mtx_gene_expression,axis=0)).tolist()
df_per_gene_summary = pd.DataFrame({'Gene_id': np.arange(len(lst_median_gene_expression)).tolist(), 'median_raw_gene_expression': lst_median_gene_expression, 'mean_raw_gene_expression': lst_mean_raw_gene_expression, 'mean_normalized_gene_expression': lst_mean_norm_gene_expression})
df_per_gene_summary.to_csv("{0}/Table_per_gene_summary_gene_expr.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)

del df_per_gene_summary

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

#create the list of unique eQTL and save beta in a table
lst_unique_eQTL = []
df_eqtl = pd.DataFrame({'Gene_id': [], 'SNP_id': [], 'Beta': []})
for current_gene_id in np.arange(6240):
    current_gene_eqtl_path = workspace_path+"/gene_id_"+str(current_gene_id)+"/refined.txt"
    if os.path.isfile(current_gene_eqtl_path):
        with open(current_gene_eqtl_path,'r') as the_file:
            the_file_lines = the_file.readlines()
            lst_eqtl_to_add = the_file_lines[0].rstrip().split(" ")
            lst_betas = the_file_lines[1].rstrip().split(" ")
            lst_unique_eQTL.extend(lst_eqtl_to_add)
            df_eqtl_to_add = pd.DataFrame({'Gene_id': [current_gene_id]*len(lst_eqtl_to_add), 'SNP_id': lst_eqtl_to_add, 'Beta': lst_betas})
            df_eqtl = pd.concat([df_eqtl, df_eqtl_to_add]).reset_index(drop=True)
lst_unique_eQTL = np.sort(np.unique(lst_unique_eQTL)).tolist()
lst_unique_eQTL = np.array(lst_unique_eQTL)[np.char.isnumeric(lst_unique_eQTL)].tolist()
lst_unique_eQTL = list(map(int, lst_unique_eQTL))
lst_unique_eQTL = [dict_raw_SNP_to_id_in_genome_mtx[raw_SNP_id] for raw_SNP_id in lst_unique_eQTL]
print("max eQTL id")
print(np.max(lst_unique_eQTL))
#Save df_eqtl
df_eqtl.to_csv("{0}/Table_ALL_eQTL.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)

del df_eqtl

#predict fitness effect size of eQTL
    # Read in the fitness data
fitnesses_data = pd.read_csv(cellranger_outs_folder+"/pheno_data_30C.txt",sep="\t",header=0,dtype={ '0': int, '1':np.float32, '2':np.float32 }).to_numpy()[:,1:3] #np.loadtxt(args.fit)
# Parse and see if it has standard errors

fitnesses = fitnesses_data[:,0]
errors = np.ones(len(fitnesses_data))

errors = np.square(errors)
errors = np.reciprocal(errors)

seed = 100000
np.random.seed(seed) # This allows us to keep the same cross validation sets.

#Identify genotypes that have finite fitness
pos_cell_to_select = np.arange(np.transpose(pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,"chr01"),sep="\t",header=None).to_numpy()).shape[1])
pos_cell_to_select = pos_cell_to_select[~np.isnan(fitnesses)]
num_usable_spores = len(pos_cell_to_select)

# Open all the genotype files
genotypes_file = []
num_lines_genotypes = []
chr_to_scan = []
base_genotypes = np.ones((num_usable_spores,1+np.sum(num_SNPs)))
current_start_id_snp = 1 # First index or index 0 is the intercept.
i = 0
for the_label_chr in lst_label_chromosomes[0:len(lst_label_chromosomes)]:
    genotypes_file.append(np.transpose(pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,the_label_chr),sep="\t",header=None).to_numpy()[pos_cell_to_select.tolist(),:]))
    #print(np.shape(genotypes_file[i])) #sanity check for genotype matrix shape
    num_lines_genotypes.append(genotypes_file[i].shape[0])
    chr_to_scan.append(i)
    base_genotypes[:,current_start_id_snp:(current_start_id_snp+num_lines_genotypes[i])] = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,the_label_chr),sep="\t",header=None).to_numpy()[pos_cell_to_select.tolist(),:]
    current_start_id_snp = current_start_id_snp + num_lines_genotypes[i]
    i = i + 1

#select eQTL
id_cols = np.concatenate((np.array([0]), (np.array(lst_unique_eQTL)+1)))
base_genotypes = base_genotypes[:,id_cols.tolist()]
#Remove genotypes that have non-finite phenotypes
fitnesses = np.array([fitnesses[i] for i in pos_cell_to_select])
errors = np.array([errors[i] for i in pos_cell_to_select])

del genotypes_file
del num_lines_genotypes
del chr_to_scan

#QR decomposition
q,r = np.linalg.qr(base_genotypes * np.sqrt(np.reshape(errors,(num_usable_spores,1))))

#Fit model and first beta index is the intercept term.
betas = linalg.solve_triangular(r,np.dot(np.transpose(q), fitnesses * np.sqrt(errors)), check_finite=False) # Beta for the WEIGHTED phenotypes.
eqtl_betas = betas[1:]
#Save eQTL betas and SNP id as the first two lines of a file
with open(workspace_path+"/eQTL_fitness_effect_sizes.txt", 'w') as f:
    # write the first list to the file
    f.write(' '.join(map(str, lst_unique_eQTL)) + '\n')
    
    # write the second list to the file
    f.write(' '.join(map(str, eqtl_betas)) + '\n') #