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
the_ind_partition = int(sys.argv[1])
nb_expression_pcs_partitions = int(sys.argv[2])
nb_expression_PCs = int(sys.argv[3])
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[4]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[5]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[6]
nb_cpus = 16 #int(sys.argv[7]) #16
cutoff_uncertainty_score = 0.2 #int(sys.argv[8])
max_ID_subset = int(sys.argv[4])

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

#matrix of corrected and imputed genotypes
try:
    df_corrected_imputed_genotypes = pd.read_csv("{0}/Not_normalized_mtx_corrected_imputed_genotypes.csv".format(workspace_path),sep="\t",header=None)
    df_corrected_imputed_genotypes.index = lst_cells
except:
    df_corrected_imputed_genotypes = pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,lst_label_chromosomes[0]),sep="\t",header=None)
    for the_label_chr in lst_label_chromosomes[1:16]:
        df_corrected_imputed_genotypes = pd.concat([df_corrected_imputed_genotypes,pd.read_csv("{0}/data/good_cells_genotypes/HMM_Genotypes_{1}.csv".format(workspace_path,the_label_chr),sep="\t",header=None)],axis=1)
    df_corrected_imputed_genotypes.index = lst_cells
    pd.DataFrame(mtx_corrected_imputed_genotypes).to_csv("{0}/Not_normalized_mtx_corrected_imputed_genotypes.csv".format(workspace_path), sep='\t',na_rep="NA",header=False,index=False)
    del mtx_corrected_imputed_genotypes

G = df_corrected_imputed_genotypes.to_numpy()
del df_corrected_imputed_genotypes
#G = np.round(G)

#matrix of gene expression (import + normalize + only select genes with finite normalized values and non-null standard deviation across cells)
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)

mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
selected_genes_in_expr_mat1 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]).tolist(),np.sum(mtx_gene_expression==0,axis=0).tolist()[0]) if j!=np.shape(mtx_gene_expression)[0] ]
backup_mtx_gene_expression = mtx_gene_expression
mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0)
selected_genes_in_expr_mat2 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]),((np.isfinite(mtx_gene_expression)).sum(axis=0).tolist())[0]) if j==np.shape(mtx_gene_expression)[0] ]
selected_genes_in_expr_mat = [int(x) for x in np.intersect1d(selected_genes_in_expr_mat1,selected_genes_in_expr_mat2)]
mtx_gene_expression = backup_mtx_gene_expression[:,selected_genes_in_expr_mat]
mtx_gene_expression = mtx_gene_expression[np.arange(max_ID_subset+1).tolist(),:]
E = mtx_gene_expression
print("Dimensions E:")
print(np.shape(E))
del mtx_gene_expression
del backup_mtx_gene_expression

#import barcodes HMM uncertainty scores
df_barcodes_uncertainty_scores = pd.read_csv("{0}/df_barcodes_uncertainty_scores.csv".format(workspace_path),sep="\t",dtype={ 'barcode': str, 'uncertainty_score':np.float32, 'rank':int })
lst_barcodes_below_max_uncertainty = (df_barcodes_uncertainty_scores[df_barcodes_uncertainty_scores["uncertainty_score"]<cutoff_uncertainty_score]).barcode.tolist()

#subsample cells to reduce execution time
    #select barcodes below uncertainty threshold
id_subsample = [index_barcode for index_barcode in np.arange(len(lst_cells)) if lst_cells[index_barcode] in lst_barcodes_below_max_uncertainty] #np.random.randint(0, np.shape(E)[0], np.shape(E)[1]).tolist()
E = E[id_subsample,:]
#subsample cells to reduce execution time
G = G[id_subsample,:]

features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
barcodes = barcodes[0:max_ID_subset]

#import phenotype data
df_pheno_30C = pd.read_csv("{0}/pheno_data_30C.txt".format(yeast_project_wp_path),sep="\t",header=0,dtype={ '0': int, '1':np.float32, '2':np.float32 })
'''
#import best matches
#df_best_match_corrected_cell_gen_vs_batch1 = pd.read_csv("{0}/onlyRMratio_filter_df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })
#nb_significant_hit_from_corrected_gen = np.sum([x < 0.05 for x in df_best_match_corrected_cell_gen_vs_batch1.pvalue.tolist() if not np.isnan(x)])
#df_best_match_org_cell_gen_vs_batch1 = pd.read_csv("{0}/onlyRMratio_filter_df_best_match_org_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })
#nb_significant_hit_from_org_gen = np.sum([x < 0.05 for x in df_best_match_org_cell_gen_vs_batch1.pvalue.tolist() if not np.isnan(x)])

#if nb_significant_hit_from_corrected_gen >= nb_significant_hit_from_org_gen:
#    df_best_matches = df_best_match_corrected_cell_gen_vs_batch1
#else:
#    df_best_matches = df_best_match_org_cell_gen_vs_batch1

#df_best_matches["fitness"] = np.nan
#id_cells_with_retrieved_fitness = []
#for i in np.arange(np.shape(df_best_matches)[0]):
#    current_pval = df_best_matches.pvalue.tolist()[i]
#    current_fitness = df_pheno_30C.iloc[df_best_matches.best_match.tolist()[i],1]
#    if (not np.isnan(current_pval)) and current_pval<0.05 and (not np.isnan(current_fitness)):
#        df_best_matches.at[i,"fitness"] = current_fitness
#        id_cells_with_retrieved_fitness.extend([i])
'''

print("End time import data:")
print(datetime.now())

#HMM-corrected genotype matrix NxM
#G = G[id_cells_with_retrieved_fitness,:]
G = (G - np.mean(G,axis=0)) / (np.sqrt(np.mean(G,axis=0)*(1-np.mean(G,axis=0)))) #G = (G - G.mean(axis=0)) / G.std(axis=0) #(pd.DataFrame(G + (1e-16)).apply(lambda x: boxcox(x)[0])).to_numpy() #
N = np.shape(G)[0]
K_G = (G @ G.T) / (np.shape(G)[1])

#free some memory
del G

'''
#Expression matrix NxL
#E = E[id_cells_with_retrieved_fitness,:]
#E = (E - E.mean(axis=0)) / E.std(axis=0) #already normalized
print("shape of expression mtx is:")
print(np.shape(E))
pca_expression_ALL_pcs = PCA(n_components=np.shape(E)[1]).fit_transform(X=E)
fit_pca_expression_ALL_pcs = PCA(n_components=np.shape(E)[1]).fit(X=E)
lst_explained_variance_ALL_pcs_expression = fit_pca_expression_ALL_pcs.explained_variance_ratio_
'''

Kernels_HMM_Genotype = [K_G]

#import expression PCA data
#nb_expression_PCs = np.shape(E)[1]
lst_starts_expression_PCs_partitions = np.arange(0,nb_expression_PCs,int(np.round(nb_expression_PCs/nb_expression_pcs_partitions)))
lst_ends_expression_PCs_partitions = np.array((lst_starts_expression_PCs_partitions[1:len(lst_starts_expression_PCs_partitions)]).tolist()+[nb_expression_PCs])
lst_starts_expression_PCs_partitions = [int(x) for x in lst_starts_expression_PCs_partitions]
lst_ends_expression_PCs_partitions = [int(x) for x in lst_ends_expression_PCs_partitions]
pca_expression_current_pcs= pd.read_csv("{0}/pca_expression_retained_pcs_{1}.csv".format(workspace_path,the_ind_partition),sep="\t",header=None).to_numpy()
pca_expression_current_pcs = pca_expression_current_pcs[id_subsample,:]
lst_explained_variance_ALL_pcs_expression = pd.read_csv("{0}/lst_explained_variance_ALL_pcs_expression.csv".format(workspace_path),sep="\t",header=None,dtype={ '0': np.float32 }).to_numpy()[:,0].tolist()
print("np.array(lst_explained_variance_ALL_pcs_expression)[0:5]:")
print(np.array(lst_explained_variance_ALL_pcs_expression)[0:5])

# Original EM REML n x n formulation
def reml_em(Kernel,X,y,sig_estimate=None,verbose=True,n_iter=60):
    # Calculating new y
    Q,R = LA.qr(X)
    M = lambda O : O - Q @ (Q.T @ O)
    resid = M(y)
    y_new = (resid - resid.mean()) / resid.std(ddof=1)

    n = Kernel[0].shape[0]
    p = len(Kernel)
    if sig_estimate is None:
        sig_estimate = np.zeros(p+1, dtype=np.float64)
        sig_estimate[:] = np.var(y_new,ddof=1) / (p+1)
    iteration_estimate = []
    # Convergence
    exit_code = 1

    for i in range(n_iter):
        prev_sig = sig_estimate.copy()

        V = sig_estimate[p] * np.eye(n, dtype=np.float64) # for I
        for j in range(p):
            V += sig_estimate[j] * Kernel[j]
        
        V_inv = LA.inv(V)
        R = V_inv - V_inv.dot(X).dot(LA.inv(X.T.dot(V_inv).dot(X))).dot(X.T).dot(V_inv)

        trace = np.zeros(p+1, dtype=np.float64)
        quad = np.zeros(p+1, dtype=np.float64)
        for j in range(p):
            quad[j] = y_new.T.dot(R).dot(Kernel[j]).dot(R).dot(y_new)
            trace[j] = np.trace( R.dot(Kernel[j]) )

        quad[p] = y_new.T.dot(R).dot(R).dot(y_new)
        trace[p] = np.trace( R )

        sig_estimate = prev_sig - ( (prev_sig**2) * (trace - quad) ) / n
        iteration_estimate.append(sig_estimate)

        print("Quad:", quad)
        print("Trace (RK):", trace)

        if verbose:
            print_str = "\t EM REML vanilla round " + str(i) + ": "
            for j in range(p + 1):
                print_str += "  {:.4f}".format(sig_estimate[j])
            print(print_str, flush=True)
        
        diff = np.max( np.abs(sig_estimate - prev_sig) )
        if diff < 1e-4 :
            exit_code = 0
            if verbose:
                print("\t Estimates converged, exitting - diff: %.6f" % (diff) )
            break

    iteration_estimate = np.array(iteration_estimate)
    return sig_estimate,iteration_estimate[0,:],iteration_estimate,exit_code


#Representation of no fixed effect
X = np.ones(N).reshape((N,1))

#initialize output array
init_mtx_output = np.zeros((np.shape(pca_expression_current_pcs)[1],4))
mtx_output = Array('d', np.shape(pca_expression_current_pcs)[1]*4)
np_mtx_output = np.frombuffer(mtx_output.get_obj()).reshape((np.shape(pca_expression_current_pcs)[1],4))
np.copyto(np_mtx_output, init_mtx_output)

def get_expression_pcs_heritability(j):
    global mtx_output

    np_mtx_output = np.frombuffer(mtx_output.get_obj()).reshape((np.shape(pca_expression_current_pcs)[1],4))

    #Get expression heritability from HMM-corrected genotypes
    current_pc_sigma_sqs_reml_HMM_genotype = reml_em(Kernel=Kernels_HMM_Genotype, X=X, y=pca_expression_current_pcs[:,j], verbose=True, n_iter=60)[0]
    np_mtx_output[j,0] = np.array(lst_explained_variance_ALL_pcs_expression)[(np.arange(lst_starts_expression_PCs_partitions[the_ind_partition],lst_ends_expression_PCs_partitions[the_ind_partition],1))[j]] #current PC explained variance ratio
    np_mtx_output[j,1] = (current_pc_sigma_sqs_reml_HMM_genotype[0])/np.sum(current_pc_sigma_sqs_reml_HMM_genotype)
    np_mtx_output[j,2] = (np.array(lst_explained_variance_ALL_pcs_expression)[(np.arange(lst_starts_expression_PCs_partitions[the_ind_partition],lst_ends_expression_PCs_partitions[the_ind_partition],1))[j]] * ((current_pc_sigma_sqs_reml_HMM_genotype[0])/np.sum(current_pc_sigma_sqs_reml_HMM_genotype)))
    np_mtx_output[j,3] = ((np.arange(lst_starts_expression_PCs_partitions[the_ind_partition],lst_ends_expression_PCs_partitions[the_ind_partition],1))[j])+1
    #if (j+1) % 10 == 0:
    print("{0} PCs analyzed out of {1}! Current sigma is {2} and var ratio PC is {3}".format(j+1,np.shape(pca_expression_current_pcs)[1],current_pc_sigma_sqs_reml_HMM_genotype[0]/np.sum(current_pc_sigma_sqs_reml_HMM_genotype),lst_explained_variance_ALL_pcs_expression[j]))

# start pool of parallel worker processes
with closing(Pool(processes=nb_cpus)) as pool:
    pool.map(get_expression_pcs_heritability, np.arange(np.shape(pca_expression_current_pcs)[1]))
    pool.terminate()

mtx_output = np_mtx_output
expression_heritability_HMM_genotype = mtx_output[:,1]

#print expression heritability
print(mtx_output)
print("current expression PC heritability estimated from HMM-corrected genotypes is {0}".format(expression_heritability_HMM_genotype))

#save expression heritability in table
data = {'Expression_heritability_HMM_genotype' : expression_heritability_HMM_genotype.tolist(),
        'id_expression_PC' : np.arange(lst_starts_expression_PCs_partitions[the_ind_partition],lst_ends_expression_PCs_partitions[the_ind_partition],1).tolist() }
df_out = pd.DataFrame(data)
df_out.to_csv("{0}/Table_HMM_based_expression_heritability_{1}.csv".format(workspace_path,the_ind_partition), sep='\t',na_rep="NA",header=True,index=False)

#save PCs explained variance, R2 and weighted R2
pd.DataFrame(mtx_output).to_csv("{0}/mtx_weighted_Expression_PCs_R2_corr_with_HMM_genotypes_{1}.csv".format(workspace_path,the_ind_partition), sep='\t',na_rep="NA",header=False,index=False)
