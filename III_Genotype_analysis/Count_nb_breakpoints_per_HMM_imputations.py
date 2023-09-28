import sys
from multiprocessing import Pool, TimeoutError, Array
from contextlib import closing
import csv
import gzip
import os
import scipy.io
from scipy.stats import boxcox
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime

print("Start time:")
print(datetime.now())

#import main workspace absolute path
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[1]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[2]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[3]
nb_cpus = 6 #int(sys.argv[4]) #16
max_ID_subset = int(sys.argv[1])

df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_lst_cells = pd.read_csv("{0}/lst_barcodes_with_expression_data.txt".format(workspace_path),sep="\t",header=None,dtype={ '0': str })
df_lst_cells = df_lst_cells.iloc[np.arange(max_ID_subset+1).tolist(),:]
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"

matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
barcodes = barcodes[0:max_ID_subset]

#matrix of corrected and imputed genotypes
'''
df_corrected_imputed_genotypes = pd.read_csv("{0}/Not_normalized_mtx_corrected_imputed_genotypes.csv".format(workspace_path),sep="\t",header=None)
df_corrected_imputed_genotypes.index = lst_cells
mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
'''
#create matrix of corrected and imputed genotypes
df_corrected_imputed_genotypes = pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,lst_label_chromosomes[0]),sep="\t",header=None)
for the_label_chr in lst_label_chromosomes[1:16]:
    df_corrected_imputed_genotypes = pd.concat([df_corrected_imputed_genotypes,pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,the_label_chr),sep="\t",header=None)],axis=1)
mtx_corrected_imputed_genotypes = df_corrected_imputed_genotypes.to_numpy()
df_corrected_imputed_genotypes = pd.DataFrame(mtx_corrected_imputed_genotypes,index=lst_cells)

arr_nb_breakpoints_HMM_imputations = np.zeros(np.shape(mtx_corrected_imputed_genotypes)[0])

for i in np.arange(np.shape(mtx_corrected_imputed_genotypes)[0]):
    current_HMM_imputation_genotype = mtx_corrected_imputed_genotypes[i,:].tolist()
    current_genotype = current_HMM_imputation_genotype[0]
    old_genotype = current_genotype
    if (current_genotype >= 0.75):
        current_allele = "RM"
    elif (current_genotype <= 0.25):
        current_allele = "BY"
    if (old_genotype >= 0.75):
        old_allele = "RM"
    elif (old_genotype <= 0.25):
        old_allele = "BY"
    for current_genotype in current_HMM_imputation_genotype:
        if (current_genotype >= 0.75):
            current_allele = "RM"
        elif (current_genotype <= 0.25):
            current_allele = "BY"
        if current_allele != old_allele:
            arr_nb_breakpoints_HMM_imputations[i] = arr_nb_breakpoints_HMM_imputations[i] + 1
        old_allele = current_allele
    print("Count of the number of breakpoints done for {0} reference lineages!".format(i+1))
data = {'barcode' : barcodes,
        'nb_breakpoints' : arr_nb_breakpoints_HMM_imputations.tolist()}
df_nb_breakpoints_HMM_imputations = pd.DataFrame(data)
df_nb_breakpoints_HMM_imputations["rank"] = df_nb_breakpoints_HMM_imputations["nb_breakpoints"].rank()
df_nb_breakpoints_HMM_imputations.to_csv("{0}/df_nb_breakpoints_HMM_imputations.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)
