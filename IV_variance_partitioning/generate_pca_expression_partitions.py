import sys
from multiprocessing import Pool, TimeoutError, Array
from contextlib import closing
import csv
import gzip
import os
import scipy.io
#from scipy.stats import boxcox
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime

print("Start time:")
print(datetime.now())

#import main workspace absolute path
nb_expression_pcs_partitions = int(sys.argv[1])
workspace_path = sys.argv[2]
cellranger_outs_folder = sys.argv[3]
nb_cpus = int(sys.argv[4]) #16
max_ID_subset = 18232

df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(workspace_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_lst_cells = pd.read_csv("{0}/lst_barcodes_with_expression_data.txt".format(workspace_path),sep="\t",header=None,dtype={ '0': str })
df_lst_cells = df_lst_cells.iloc[np.arange(max_ID_subset+1).tolist(),:]
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"

#matrix of gene expression (import + normalize + only select genes with finite normalized values and non-null standard deviation across cells)
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
backup_mtx_gene_expression = mtx_gene_expression
selected_genes_in_expr_mat1 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]).tolist(),np.sum(mtx_gene_expression==0,axis=0).tolist()[0]) if j!=np.shape(mtx_gene_expression)[0] ]
mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0)
selected_genes_in_expr_mat2 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]),((np.isfinite(mtx_gene_expression)).sum(axis=0).tolist())[0]) if j==np.shape(mtx_gene_expression)[0] ]
selected_genes_in_expr_mat = [int(x) for x in np.intersect1d(selected_genes_in_expr_mat1,selected_genes_in_expr_mat2)]
mtx_gene_expression = backup_mtx_gene_expression[:,selected_genes_in_expr_mat]
mtx_gene_expression = mtx_gene_expression[np.arange(max_ID_subset+1).tolist(),:]
E = mtx_gene_expression
#free some memory
del mtx_gene_expression
del backup_mtx_gene_expression

#import phenotype data
df_pheno_30C = pd.read_csv("{0}/pheno_data_30C.txt".format(workspace_path),sep="\t",header=0,dtype={ '0': int, '1':np.float32, '2':np.float32 })

features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
barcodes = barcodes[0:max_ID_subset]

print("End time import data:")
print(datetime.now())

#Expression matrix NxL
#E = E[id_cells_with_retrieved_fitness,:]
print("shape of expression mtx is:")
print(np.shape(E))
pca_expression_ALL_pcs = PCA(n_components=np.shape(E)[1]).fit_transform(X=E)
fit_pca_expression_ALL_pcs = PCA(n_components=np.shape(E)[1]).fit(X=E)
lst_explained_variance_ALL_pcs_expression = fit_pca_expression_ALL_pcs.explained_variance_ratio_
pd.DataFrame(pca_expression_ALL_pcs).to_csv("{0}/pca_expression_ALL_pcs.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)

#save lst_explained_variance_ALL_pcs_expression in table
data = {'lst_explained_variance_ALL_pcs_expression' : lst_explained_variance_ALL_pcs_expression}
df_out = pd.DataFrame(data)
df_out.to_csv("{0}/lst_explained_variance_ALL_pcs_expression.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)

#find the number of PCs explaining 99% of the variance
cumul_var_expl = 0
nb_expression_PCs = 1
while cumul_var_expl < 0.99:
    cumul_var_expl = cumul_var_expl + lst_explained_variance_ALL_pcs_expression[nb_expression_PCs-1]
    nb_expression_PCs = nb_expression_PCs + 1
print("Number of expression PCs saved = {0}".format(nb_expression_PCs))

#Create and save expression PCs partitions
#nb_expression_PCs = np.shape(E)[1]
lst_starts_expression_PCs_partitions = np.arange(0,nb_expression_PCs,np.round(nb_expression_PCs/nb_expression_pcs_partitions))
lst_ends_expression_PCs_partitions = np.array((lst_starts_expression_PCs_partitions[1:len(lst_starts_expression_PCs_partitions)]).tolist()+[nb_expression_PCs-1]).tolist()
lst_starts_expression_PCs_partitions = lst_starts_expression_PCs_partitions.tolist()
lst_starts_expression_PCs_partitions = [int(x) for x in lst_starts_expression_PCs_partitions]
lst_ends_expression_PCs_partitions = [int(x) for x in lst_ends_expression_PCs_partitions]
print(lst_starts_expression_PCs_partitions)
print(lst_ends_expression_PCs_partitions)
for ind_partition in np.arange(nb_expression_pcs_partitions):
    pd.DataFrame(pca_expression_ALL_pcs[:,(lst_starts_expression_PCs_partitions[ind_partition]):(lst_ends_expression_PCs_partitions[ind_partition])]).to_csv("{0}/pca_expression_retained_pcs_{1}.csv".format(workspace_path,ind_partition), sep='\t',na_rep="NA",header=False,index=False)
