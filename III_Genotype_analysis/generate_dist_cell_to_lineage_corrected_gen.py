#df_author=Arnaud NG
#Script for correcting genotypes using HMMs 

#import libraries
import sys
import time
from multiprocessing import Pool, TimeoutError, Array
from contextlib import closing
import csv
import gzip
import os
import scipy.io
from scipy.stats import mannwhitneyu, boxcox #, linregress
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from random import sample, seed
from math import factorial, log, exp
import matplotlib
matplotlib.use('Agg')
from matplotlib import patches
import matplotlib.pyplot as plt
import seaborn as sns
import random
import timeit
from datetime import datetime
#from scipy.spatial.distance import cdist

print("Start time:")
print(datetime.now())

#import main workspace absolute path
workspace_path = sys.argv[1]
yeast_project_wp_path = sys.argv[2]
cellranger_outs_folder = sys.argv[3]
nb_cpus = int(sys.argv[4]) #16
nb_subsampling = int(sys.argv[5]) #500

df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_lst_cells = pd.read_csv("{0}/lst_barcodes_with_expression_data.txt".format(workspace_path),sep="\t",header=None,dtype={ '0': str })
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"
'''
#import set of files that have already been concatenated
os.system("rm -f {0}/data/good_cells_genotypes/All_chromosomes_corrected_MOCKFILE.csv".format(workspace_path))
os.system("touch {0}/data/good_cells_genotypes/All_chromosomes_corrected_MOCKFILE.csv".format(workspace_path))
try:
    os.system("ls {0}/data/good_cells_genotypes/All_chromosomes_corrected_*.csv > {0}/lst_concatenated_corrected_genotypes.txt".format(workspace_path))
except:
    print("")

set_already_concatenated_corr_gen = set(pd.read_csv("{0}/lst_concatenated_corrected_genotypes.txt".format(workspace_path),sep="\t",header=None).iloc[:,0].tolist())

#matrix of corrected and imputed genotypes
init_mtx_corrected_imputed_genotypes = np.zeros((nb_cells,np.shape(df_pos_snps)[0]))
mtx_corrected_imputed_genotypes = Array('d', nb_cells*np.shape(df_pos_snps)[0])
np_mtx_corrected_imputed_genotypes = np.frombuffer(mtx_corrected_imputed_genotypes.get_obj()).reshape((nb_cells,np.shape(df_pos_snps)[0]))
np.copyto(np_mtx_corrected_imputed_genotypes, init_mtx_corrected_imputed_genotypes)
    
def populate_mtx_corrected_imputed_genotypes(id_cell):
    global mtx_corrected_imputed_genotypes
    current_cell = lst_cells[id_cell]
    np_mtx_corrected_imputed_genotypes = np.frombuffer(mtx_corrected_imputed_genotypes.get_obj()).reshape((nb_cells,np.shape(df_pos_snps)[0]))
    #delete header in all corrected_Genotype_... of current cell, cat them, import and add column name to pandas dataframe
    name_file = "{0}/data/good_cells_genotypes/All_chromosomes_corrected_imputed_Genotype_table_Cell_CB_Z_{1}.csv".format(workspace_path,current_cell)
    if not name_file in set_already_concatenated_corr_gen:
        os.system("cat {0}/data/good_cells_genotypes/corrected_imputed_Genotype_table_*_Cell_CB_Z_{1}.csv | grep -v RM > {0}/data/good_cells_genotypes/All_chromosomes_corrected_imputed_Genotype_table_Cell_CB_Z_{1}.csv".format(workspace_path,current_cell))
    df_current_cell_corrected_gen = pd.read_csv("{0}/data/good_cells_genotypes/All_chromosomes_corrected_imputed_Genotype_table_Cell_CB_Z_{1}.csv".format(workspace_path,current_cell),sep="\t",header=None)
    df_current_cell_corrected_gen.columns = ["BY","RM","RM_ratio","chromosome","position","the_key","corrected_genotypes"]
    df_current_cell_corrected_gen= df_current_cell_corrected_gen.astype({ 'BY': int, 'RM': int, 'RM_ratio': np.float32, 'chromosome': str, 'position' : int, 'the_key' : str, 'corrected_genotypes':np.float32 })
    np_mtx_corrected_imputed_genotypes[id_cell] = df_current_cell_corrected_gen.corrected_genotypes.tolist()
    if id_cell%1000==0:
        print("Imputed genotype retrieved for cell {0} out of {1}!".format(id_cell,nb_cells))

# start pool of parallel worker processes
with closing(Pool(processes=16)) as pool:
    pool.map(populate_mtx_corrected_imputed_genotypes, np.arange(nb_cells))
    pool.terminate()

mtx_corrected_imputed_genotypes = np_mtx_corrected_imputed_genotypes
df_corrected_imputed_genotypes = pd.DataFrame(mtx_corrected_imputed_genotypes,index=lst_cells)
'''

#mtx_corrected_imputed_genotypes = pd.read_csv("{0}/Not_normalized_mtx_corrected_imputed_genotypes.csv".format(workspace_path),sep="\t",header=None).to_numpy()

#create matrix of corrected and imputed genotypes
df_corrected_imputed_genotypes = pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,lst_label_chromosomes[0]),sep="\t",header=None)
for the_label_chr in lst_label_chromosomes[1:16]:
    df_corrected_imputed_genotypes = pd.concat([df_corrected_imputed_genotypes,pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,the_label_chr),sep="\t",header=None)],axis=1)
mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
df_corrected_imputed_genotypes = pd.DataFrame(mtx_corrected_imputed_genotypes,index=lst_cells)

print("corrected genotype matrix loaded:")
print(datetime.now())

#matrix of gene expression
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
    #ONLY SELECT GENES THAT ARE EXPRESSED IN AT LEAST 1 CELL
selected_genes_in_expr_mat = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]),np.array((mtx_gene_expression==0).sum(axis=0))[0]) if j!=np.shape(mtx_gene_expression)[0] ]
mtx_gene_expression = mtx_gene_expression[:,selected_genes_in_expr_mat]
pd.DataFrame(mtx_gene_expression).to_csv("{0}/Not_normalized_mtx_gene_expression.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)

features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
'''
#make sure that mtx_corrected_imputed_genotypes and mtx_gene_expression have the same rows
#df_corrected_imputed_genotypes = df_corrected_imputed_genotypes.reindex(barcodes)
#mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
#mtx_corrected_imputed_genotypes = np.round(np_mtx_corrected_imputed_genotypes)

pd.DataFrame(mtx_corrected_imputed_genotypes).to_csv("{0}/Not_normalized_mtx_corrected_imputed_genotypes.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)
'''
#import phenotype data
df_pheno_30C = pd.read_csv("{0}/pheno_data_30C.txt".format(yeast_project_wp_path),sep="\t",header=0,dtype={ '0': int, '1':np.float32, '2':np.float32 })

#import ALL genotypes
df_all_reference_lineage_genotypes = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,lst_label_chromosomes[0]),sep="\t",header=None)

for the_label_chr in lst_label_chromosomes[1:len(lst_label_chromosomes)]:
    df_to_add = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,the_label_chr),sep="\t",header=None)
    df_to_add = df_to_add.reset_index(drop=True)
    df_all_reference_lineage_genotypes = pd.concat([df_all_reference_lineage_genotypes,df_to_add],axis=1)
mtx_all_reference_lineage_genotypes = df_all_reference_lineage_genotypes.to_numpy()
#mtx_all_reference_lineage_genotypes = np.round(mtx_all_reference_lineage_genotypes)
'''
#Polymorphic sites uncertainty score (calculated from mtx_corrected_imputed_genotypes)
arr_HMM_polsites_uncertainty_scores = np.zeros(np.shape(mtx_corrected_imputed_genotypes)[1])
for ind_HMM_polsite in np.arange(np.shape(mtx_corrected_imputed_genotypes)[1]):
    lst_nonan_genotype_current_cell = (mtx_corrected_imputed_genotypes[:,ind_HMM_polsite])
    arr_HMM_polsites_uncertainty_scores[ind_HMM_polsite] = (4/len(lst_nonan_genotype_current_cell))*np.sum(lst_nonan_genotype_current_cell*(1-(lst_nonan_genotype_current_cell)))
#save scores
data = {'SNP_id' : np.arange(np.shape(mtx_corrected_imputed_genotypes)[1]).tolist(),
        'uncertainty_score' : arr_HMM_polsites_uncertainty_scores.tolist()}
df_HMM_polsites_uncertainty_scores = pd.DataFrame(data)
df_HMM_polsites_uncertainty_scores["rank"] = df_HMM_polsites_uncertainty_scores["uncertainty_score"].rank()
df_HMM_polsites_uncertainty_scores.to_csv("{0}/df_HMM_polsites_uncertainty_scores.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)

#Polymorphic sites uncertainty score (calculated from mtx_all_reference_lineage_genotypes)
arr_ref_dataset_polsites_uncertainty_scores = np.zeros(np.shape(mtx_all_reference_lineage_genotypes)[1])
for ind_ref_dataset_polsite in np.arange(np.shape(mtx_all_reference_lineage_genotypes)[1]):
    lst_nonan_genotype_current_cell = (mtx_all_reference_lineage_genotypes[:,ind_ref_dataset_polsite])
    arr_ref_dataset_polsites_uncertainty_scores[ind_ref_dataset_polsite] = (4/len(lst_nonan_genotype_current_cell))*np.sum(lst_nonan_genotype_current_cell*(1-(lst_nonan_genotype_current_cell)))
#save scores
data = {'SNP_id' : np.arange(np.shape(mtx_all_reference_lineage_genotypes)[1]).tolist(),
        'uncertainty_score' : arr_ref_dataset_polsites_uncertainty_scores.tolist()}
df_ref_dataset_polsites_uncertainty_scores = pd.DataFrame(data)
df_ref_dataset_polsites_uncertainty_scores["rank"] = df_ref_dataset_polsites_uncertainty_scores["uncertainty_score"].rank()
df_ref_dataset_polsites_uncertainty_scores.to_csv("{0}/df_ref_dataset_polsites_uncertainty_scores.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)

#save metadata about selected expressed genes names
data = {'Expressed_gene_original_index' : selected_genes_in_expr_mat,
        'Expressed_gene_index_in_concat_mtx' : np.arange(len(selected_genes_in_expr_mat))+(np.shape(mtx_corrected_imputed_genotypes)[1]),
        'Gene_name':np.array(gene_names)[selected_genes_in_expr_mat].tolist()}
pd.DataFrame(data).to_csv("{0}/df_expressed_genes_selected.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)
'''
print("End time import data:")
print(datetime.now())

#Computes the euclidean distance between each pair of genotypes (corrected cell genotype vs ALL reference lineages genotype)
    #only use position with certainty for lineage assignement

filtered_mtx_corrected_RM_ratio = mtx_corrected_imputed_genotypes
filtered_mtx_all_reference_lineage_genotypes = mtx_all_reference_lineage_genotypes
#filtered_mtx_corrected_RM_ratio[(abs(filtered_mtx_corrected_RM_ratio-0.5)<0.25)] = np.nan
#filtered_mtx_all_reference_lineage_genotypes[(abs(filtered_mtx_all_reference_lineage_genotypes-0.5)<0.25)] = np.nan

del mtx_all_reference_lineage_genotypes

try:
    mtx_dist_corrected_genotype_vs_all_reference_lineages= pd.read_csv("{0}/mtx_dist_corrected_genotype_vs_all_reference_lineages.csv".format(workspace_path),sep="\t",header=None).to_numpy()
except:
    #weighted Euclidean
    #mtx_dist_corrected_genotype_vs_all_reference_lineages = cdist(filtered_mtx_corrected_RM_ratio, filtered_mtx_all_reference_lineage_genotypes, 'euclidean', w = ((1-arr_HMM_polsites_uncertainty_scores)*(1-arr_ref_dataset_polsites_uncertainty_scores)))  #nan_euclidean_distances(filtered_mtx_corrected_RM_ratio, filtered_filtered_mtx_all_reference_lineage_genotypes)
    #Euclidean
    #mtx_dist_corrected_genotype_vs_all_reference_lineages = euclidean_distances(filtered_mtx_corrected_RM_ratio, filtered_mtx_all_reference_lineage_genotypes)
    #calculate expected distance
    mtx_expected_dist_component_from_HMM_cells_only = np.repeat(np.sum(filtered_mtx_corrected_RM_ratio,axis=1).reshape(filtered_mtx_corrected_RM_ratio.shape[0],1),filtered_mtx_all_reference_lineage_genotypes.shape[0],axis=1)
    mtx_expected_dist_component_from_HMM_refs_only = np.repeat(np.sum(filtered_mtx_all_reference_lineage_genotypes,axis=1).reshape(1,filtered_mtx_all_reference_lineage_genotypes.shape[0]),filtered_mtx_corrected_RM_ratio.shape[0],axis=0)
    mtx_dist_corrected_genotype_vs_all_reference_lineages = mtx_expected_dist_component_from_HMM_cells_only + mtx_expected_dist_component_from_HMM_refs_only - (2*np.dot(filtered_mtx_corrected_RM_ratio,np.transpose(filtered_mtx_all_reference_lineage_genotypes)))
    pd.DataFrame(mtx_dist_corrected_genotype_vs_all_reference_lineages).to_csv("{0}/mtx_dist_corrected_genotype_vs_all_reference_lineages.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)

#initialize dataframe of best matches in batch1
nb_lineages_in_batch1 = np.shape(pd.read_csv("{0}/{1}_spore_major_batch_1.txt".format(yeast_project_wp_path,lst_label_chromosomes[0]),sep="\t",header=None).to_numpy())[0]
df_best_match_corrected_cell_gen_vs_batch1 = pd.DataFrame(data={'best_match' : np.nanargmin(mtx_dist_corrected_genotype_vs_all_reference_lineages[:,0:nb_lineages_in_batch1],axis=1), 'min_dist': np.nanmin(mtx_dist_corrected_genotype_vs_all_reference_lineages[:,0:nb_lineages_in_batch1],axis=1), 'pvalue' : np.nan},index=lst_cells)

init_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score = np.zeros((nb_cells,nb_subsampling))
mtx_corrected_gen_vs_independent_batches_random_subsample_min_score = Array('d', nb_cells*nb_subsampling)
np_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score = np.frombuffer(mtx_corrected_gen_vs_independent_batches_random_subsample_min_score.get_obj()).reshape((nb_cells,nb_subsampling))
np.copyto(np_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score, init_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score)

    #p-values computation
def corrected_genotype_vs_subsamples_lineage_assignment(id_subsample):
    global mtx_corrected_gen_vs_independent_batches_random_subsample_min_score
    np_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score = np.frombuffer(mtx_corrected_gen_vs_independent_batches_random_subsample_min_score.get_obj()).reshape((nb_cells,nb_subsampling))
    #random lineages from other batches
    random.seed(id_subsample)

    #get distances to a random subsample from other batches
    nb_rows_all_genotype_df = np.shape(pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,lst_label_chromosomes[0]),sep="\t",header=None).to_numpy())[0]
    mtx_dist_corrected_genotype_vs_subsample = mtx_dist_corrected_genotype_vs_all_reference_lineages[:,np.random.randint(4489, nb_rows_all_genotype_df, 4489).tolist()]
    np_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score[:,id_subsample] = pd.DataFrame(mtx_dist_corrected_genotype_vs_subsample).min(axis=1)
    if id_subsample%nb_cpus==0:
        print("Distances between corrected genotype and random reference genotypes from subsample {0} computed (out of {1} subsamples)!".format(id_subsample+1,nb_subsampling))

# start pool of parallel worker processes
with closing(Pool(processes=nb_cpus)) as pool:
    pool.map(corrected_genotype_vs_subsamples_lineage_assignment, np.arange(nb_subsampling))
    pool.terminate()
mtx_corrected_gen_vs_independent_batches_random_subsample_min_score = np_mtx_corrected_gen_vs_independent_batches_random_subsample_min_score

init_arr_best_matches_corrected_genotype_vs_subsample_pvals = np.ones((nb_cells,1))
arr_best_matches_corrected_genotype_vs_subsample_pvals = Array('d', nb_cells)
np_arr_best_matches_corrected_genotype_vs_subsample_pvals = np.frombuffer(arr_best_matches_corrected_genotype_vs_subsample_pvals.get_obj()).reshape((nb_cells,1))
np.copyto(np_arr_best_matches_corrected_genotype_vs_subsample_pvals, init_arr_best_matches_corrected_genotype_vs_subsample_pvals)

def get_pvals_corrected_genotype_vs_subsamples_lineage_assignment(id_cell):
    global arr_best_matches_corrected_genotype_vs_subsample_pvals
    np_arr_best_matches_corrected_genotype_vs_subsample_pvals = np.frombuffer(arr_best_matches_corrected_genotype_vs_subsample_pvals.get_obj()).reshape((nb_cells,1))

    #original distance with best hit
    v_org_min_dist = df_best_match_corrected_cell_gen_vs_batch1.min_dist.tolist()[id_cell]
    np_arr_best_matches_corrected_genotype_vs_subsample_pvals[id_cell,0] = np.nansum(mtx_corrected_gen_vs_independent_batches_random_subsample_min_score[id_cell] <= v_org_min_dist)/(nb_subsampling)
    if id_cell%200==0:
        print("Min dist for cell {0} is {1} and p-value is {2}".format(lst_cells[id_cell],v_org_min_dist,np_arr_best_matches_corrected_genotype_vs_subsample_pvals[id_cell,0]))
        print("P-value lineage assignment computed for corrected genotype of cell {0} out of {1}!".format(id_cell+1,np.shape(df_best_match_corrected_cell_gen_vs_batch1)[0]))
# start pool of parallel worker processes
with closing(Pool(processes=nb_cpus)) as pool:
    pool.map(get_pvals_corrected_genotype_vs_subsamples_lineage_assignment, np.arange(np.shape(df_best_match_corrected_cell_gen_vs_batch1)[0]))
    pool.terminate()

arr_best_matches_corrected_genotype_vs_subsample_pvals = np_arr_best_matches_corrected_genotype_vs_subsample_pvals

for i in np.arange(np.shape(arr_best_matches_corrected_genotype_vs_subsample_pvals)[0]):
    df_best_match_corrected_cell_gen_vs_batch1.at[lst_cells[i],"pvalue"] = arr_best_matches_corrected_genotype_vs_subsample_pvals[i,0]

df_best_match_corrected_cell_gen_vs_batch1.to_csv("{0}/df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)
print("P-values of lineage assignment for corrected genotypes computed!")
print("End time of the computation of the Euclidean distances (Corrected vs batch1):")
print(datetime.now())