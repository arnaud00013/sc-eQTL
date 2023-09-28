import sys
import csv
import gzip
import os
import scipy.io
#from scipy.stats import boxcox
from multiprocessing import Pool, TimeoutError, Array
from contextlib import closing
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
nb_cpus = 32 #int(sys.argv[4]) #16
cutoff_uncertainty_score = 0.2 #int(sys.argv[5]) #ONLY USED WHEN GENOTYPE IS THE HMM IMPUTATION
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

#matrix of corrected and imputed genotypes
df_corrected_imputed_genotypes = pd.read_csv("{0}/Not_normalized_mtx_corrected_imputed_genotypes.csv".format(workspace_path),sep="\t",header=None)
df_corrected_imputed_genotypes.index = lst_cells
G = df_corrected_imputed_genotypes.to_numpy()

del df_corrected_imputed_genotypes

#import phenotype data
df_pheno_30C = pd.read_csv("{0}/pheno_data_30C.txt".format(yeast_project_wp_path),sep="\t",header=0,dtype={ '0': int, '1':np.float32, '2':np.float32 })

#import best matches
#df_best_match_corrected_cell_gen_vs_batch1 = pd.read_csv("{0}/df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })
#nb_significant_hit_from_corrected_gen = np.sum([x < 0.05 for x in df_best_match_corrected_cell_gen_vs_batch1.pvalue.tolist() if not np.isnan(x)])
#df_best_match_org_cell_gen_vs_batch1 = pd.read_csv("{0}/df_best_match_org_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })
#nb_significant_hit_from_org_gen = np.sum([x < 0.05 for x in df_best_match_org_cell_gen_vs_batch1.pvalue.tolist() if not np.isnan(x)])

#if nb_significant_hit_from_corrected_gen >= nb_significant_hit_from_org_gen:
#    df_best_matches = df_best_match_corrected_cell_gen_vs_batch1
#else:
#    df_best_matches = df_best_match_org_cell_gen_vs_batch1


#df_best_matches = pd.read_csv("{0}/backup_expected_distance_df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })
df_best_matches = pd.read_csv("{0}/df_best_match_org_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ 'best_match': int, 'min_dist':np.float32, 'pvalue':np.float32 })

df_best_matches["fitness"] = np.nan
id_cells_with_retrieved_fitness = []
for i in np.arange(np.shape(df_best_matches)[0]):
    current_pval = df_best_matches.pvalue.tolist()[i]
    current_fitness = df_pheno_30C.iloc[df_best_matches.best_match.tolist()[i],1]
    if (not np.isnan(current_pval)) and current_pval<0.05 and (not np.isnan(current_fitness)):
        df_best_matches.at[i,"fitness"] = current_fitness
        id_cells_with_retrieved_fitness.extend([i])


#import barcodes HMM uncertainty scores
df_barcodes_uncertainty_scores = pd.read_csv("{0}/df_barcodes_uncertainty_scores.csv".format(workspace_path),sep="\t",dtype={ 'barcode': str, 'uncertainty_score':np.float32, 'rank':int })
lst_barcodes_below_max_uncertainty = (df_barcodes_uncertainty_scores[df_barcodes_uncertainty_scores["uncertainty_score"]<cutoff_uncertainty_score]).barcode.tolist()
    #select barcodes below uncertainty threshold
id_subsample = [index_barcode for index_barcode in np.arange(len(lst_cells)) if lst_cells[index_barcode] in lst_barcodes_below_max_uncertainty] #np.random.randint(0, np.shape(E)[0], np.shape(E)[1]).tolist()
id_cells_with_retrieved_fitness_and_enough_certainty = np.intersect1d(id_cells_with_retrieved_fitness,id_subsample)


df_best_matches = df_best_matches.reset_index(drop=True)
print(df_best_matches.iloc[0:10,:])


print("id_cells_with_retrieved_fitness:")
print(id_cells_with_retrieved_fitness)
print("id_subsample:")
print(id_subsample)
print("id_cells_with_retrieved_fitness_and_enough_certainty:")
print(id_cells_with_retrieved_fitness_and_enough_certainty)

#HMM-corrected genotype matrix NxM
G = G[id_cells_with_retrieved_fitness_and_enough_certainty,:] #G[id_cells_with_retrieved_fitness,:]
G = (G - np.mean(G,axis=0)) / (np.sqrt(np.mean(G,axis=0)*(1-np.mean(G,axis=0)))) #(G - G.mean(axis=0)) / G.std(axis=0) #(pd.DataFrame(G + (1e-16)).apply(lambda x: boxcox(x)[0])).to_numpy() #
print("np.shape(G):")
print(np.shape(G))
'''
pca_loadings_genotype = PCA(n_components=G.shape[0]).fit_transform(X=G)
fit_pca_genotype_only = PCA(n_components=G.shape[0]).fit(X=G)
lst_explained_variance_genotype_pcs = fit_pca_genotype_only.explained_variance_ratio_
cumul = 0
for i in np.arange(len(lst_explained_variance_genotype_pcs)):
    cumul = cumul + lst_explained_variance_genotype_pcs[i]
    if cumul >= 0.99:
        nb_pcs_to_retain = i + 1
        break

G = pca_loadings_genotype[:,0:nb_pcs_to_retain]
'''
K_G = (G @ G.T) / (np.shape(G)[1])

'''
#import ALL reference lineage genotypes
df_all_reference_lineage_genotypes = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,lst_label_chromosomes[0]),sep="\t",header=None)

for the_label_chr in lst_label_chromosomes[1:len(lst_label_chromosomes)]:
    df_to_add = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,the_label_chr),sep="\t",header=None)
    df_to_add = df_to_add.reset_index(drop=True)
    df_all_reference_lineage_genotypes = pd.concat([df_all_reference_lineage_genotypes,df_to_add],axis=1)
mtx_all_reference_lineage_genotypes = df_all_reference_lineage_genotypes.to_numpy()
#mtx_all_reference_lineage_genotypes = np.round(mtx_all_reference_lineage_genotypes)
#Reference lineage genotype matrix
ref_G = mtx_all_reference_lineage_genotypes[df_best_matches.best_match.tolist(),:]
#select cells with fitness data (significant association to a reference lineage)
ref_G = ref_G[id_cells_with_retrieved_fitness,:]
#average the expression of reference lineages that are assigned more than once and only select the genotype once to avoid singular matrix (non-invertible)
lst_assigned_lineages = np.array(df_best_matches.best_match.tolist())[id_cells_with_retrieved_fitness].tolist()
lst_not_duplicated = []
for current_lin in lst_assigned_lineages:
    if lst_assigned_lineages.count(current_lin) > 1:
        lst_not_duplicated.append(False)
    else:
        lst_not_duplicated.append(True)
ref_G = ref_G[lst_not_duplicated,:]
'''

#matrix of gene expression (import + normalize + only select genes with finite normalized values and non-null standard deviation across cells)
matrix_dir = "{0}/filtered_feature_bc_matrix".format(cellranger_outs_folder)
mtx_gene_expression = (scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))).transpose().todense()
#backup_mtx_gene_expression = mtx_gene_expression
selected_genes_in_expr_mat1 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]).tolist(),np.sum(mtx_gene_expression==0,axis=0).tolist()[0]) if j!=np.shape(mtx_gene_expression)[0] ]
mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0)
selected_genes_in_expr_mat2 = [i for (i,j) in zip(np.arange(np.shape(mtx_gene_expression)[1]).tolist(),((np.isfinite(mtx_gene_expression)).sum(axis=0).tolist())[0]) 
if j==np.shape(mtx_gene_expression)[0] ]
selected_genes_in_expr_mat = np.intersect1d(selected_genes_in_expr_mat1,selected_genes_in_expr_mat2).tolist()
mtx_gene_expression = mtx_gene_expression[:,selected_genes_in_expr_mat]  #backup_mtx_gene_expression[:,selected_genes_in_expr_mat]
mtx_gene_expression = mtx_gene_expression[np.arange(max_ID_subset+1).tolist(),:]
E = mtx_gene_expression

#Expression matrix NxL
E = E[id_cells_with_retrieved_fitness_and_enough_certainty,:] #E[id_cells_with_retrieved_fitness,:]
'''
#expression matrix with average for barcodes with same reference lineage assignment
new_E = np.zeros((np.sum(lst_not_duplicated),np.shape(E)[1]))
for ind_row in np.arange(np.shape(new_E)[0]):
    if lst_not_duplicated[ind_row]:
        new_E[ind_row,:] = E[ind_row,:]
    else:
        lst_indexes_barcodes_with_current_lineage = lst_assigned_lineages.index(lst_assigned_lineages[ind_row])
        new_E[ind_row,:] = np.nanmean(E[lst_indexes_barcodes_with_current_lineage,:], axis=0)
E = new_E #E[lst_not_duplicated,:]
E = (E - np.nanmean(E,axis=0))/np.nanstd(E,axis=0)
sum_gene = (np.isfinite(E).sum(axis=0).tolist())
selected_genes_in_expr_mat_after_mean_expr_calc_for_duplicates = []
for j in np.arange(np.shape(E)[1]):
    if (sum_gene[j]) == (np.shape(E)[0]):
        selected_genes_in_expr_mat_after_mean_expr_calc_for_duplicates.append(j)
#selected_genes_in_expr_mat_after_mean_expr_calc_for_duplicates = [i for (i,j) in zip(np.arange(np.shape(E)[1]).tolist(),(np.isfinite(E).sum(axis=0).tolist())[0])if j==np.shape(E)[0] ]
E = E[:,selected_genes_in_expr_mat_after_mean_expr_calc_for_duplicates]
print("E after normalization and after gene selection")
print(np.shape(E))
print(E[0:10,0:10])
'''
features_path = os.path.join(matrix_dir, "features.tsv.gz")
feature_ids = [row[0] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
gene_names = [row[1] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
feature_types = [row[2] for row in csv.reader(gzip.open(features_path,'rt'), delimiter="\t")]
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path,'rt'), delimiter="\t")]
barcodes = barcodes[0:max_ID_subset]

print("End time import data:")
print(datetime.now())

#Fitness 
#select cells with fitness data (significant association to a reference lineage)
y = np.array([df_best_matches.fitness.tolist()[z] for z in id_cells_with_retrieved_fitness_and_enough_certainty]) #np.array([df_best_matches.fitness.tolist()[z] for z in id_cells_with_retrieved_fitness])

#normalize fitness data
y = (y - np.nanmean(y))/np.nanstd(y)
#y = y[lst_not_duplicated]
#remove lineages without fitness data
current_mask = ~np.isnan(y)
y = y[current_mask]

G = G[current_mask,:]
N = np.shape(G)[0]

'''
ref_G = ref_G[current_mask,:]
N = np.shape(ref_G)[0]
ref_G = (ref_G - ref_G.mean(axis=0)) / ref_G.std(axis=0) #(pd.DataFrame(ref_G + (1e-16)).apply(lambda x: boxcox(x)[0])).to_numpy() #
pca_loadings_genotype = PCA(n_components=ref_G.shape[0]).fit_transform(X=ref_G)
fit_pca_genotype_only = PCA(n_components=ref_G.shape[0]).fit(X=ref_G)
lst_explained_variance_genotype_pcs = fit_pca_genotype_only.explained_variance_ratio_
cumul = 0
for i in np.arange(len(lst_explained_variance_genotype_pcs)):
    cumul = cumul + lst_explained_variance_genotype_pcs[i]
    if cumul >= 0.99:
        nb_pcs_to_retain = i + 1
        break

ref_G = pca_loadings_genotype[:,0:nb_pcs_to_retain]

K_G = (ref_G @ ref_G.T) / (np.shape(ref_G)[1])
'''

E = E[current_mask,:]
print("E after nan fitness mask")
print(np.shape(E))
print(E[0:10,0:10])
K_E = (E @ E.T) / (np.shape(E)[1])
print(K_E[0:5,0:5])

Kernels_Genotype_only = [K_G]
Kernels_Expression_only = [K_E]
Kernels_G_and_E = [K_G, K_E]

# Original EI REML n x n formulation
def reml_ei(Kernel,X,y,verbose=True,n_iter=200):
    n = Kernel[0].shape[0]
    p = len(Kernel)
    sig_estimate = np.zeros(p+1, dtype=np.float64)
    sig_estimate[:] = np.var(y,ddof=1) / (p+1)
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
        
        Information = np.zeros((p+1,p+1), dtype=np.float64)
        score = np.zeros(p+1, dtype=np.float64)
        for j in range(p):
            score[j] = np.trace( R.dot(Kernel[j]) ) - y.T.dot(R).dot(Kernel[j]).dot(R).dot(y)
            Information[j,p] = np.trace( R.dot(Kernel[j]).dot(R) ) 
            Information[p,j] = Information[j,p]
            for k in range(j,p):
                Information[j,k] = np.trace( R.dot(Kernel[j]).dot(R).dot(Kernel[k]) )
                Information[k,j] = Information[j,k]
        
        Information[p,p] = np.trace( R.dot(R) )
        Information = 0.5 * Information
        score[p] = np.trace(R) - y.T.dot(R).dot(R).dot(y)
        score = -0.5 * score
        
        delta = LA.inv(Information).dot(score)
        sig_estimate = sig_estimate + delta
        iteration_estimate.append(sig_estimate)

        print("Information:", Information)
        print("Score:", score)

        if verbose:
            print_str = "\t EI REML vanilla round " + str(i) + ": "
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


# Original EM REML n x n formulation
def reml_em(Kernel,X,y,sig_estimate=None,verbose=True,n_iter=200):
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

#Get genotype and expression explained variance (vs Phenotype)
sigma_sqs_reml_Pheno_vs_G_and_interaction = reml_em(Kernel=Kernels_Genotype_only, sig_estimate = np.array([0.5,0.5]), X=X, y=y, verbose=True, n_iter=200)[0]
variance_ratio_explained_by_G_and_interaction = (sigma_sqs_reml_Pheno_vs_G_and_interaction[0])/np.sum(sigma_sqs_reml_Pheno_vs_G_and_interaction)
print(sigma_sqs_reml_Pheno_vs_G_and_interaction)

sigma_sqs_reml_Pheno_vs_E_and_interaction = reml_em(Kernel=Kernels_Expression_only, sig_estimate = np.array([0.5,0.5]), X=X, y=y, verbose=True, n_iter=200)[0]
variance_ratio_explained_by_E_and_interaction = (sigma_sqs_reml_Pheno_vs_E_and_interaction[0])/np.sum(sigma_sqs_reml_Pheno_vs_E_and_interaction)
print(sigma_sqs_reml_Pheno_vs_E_and_interaction)

sigma_sqs_reml_Pheno_vs_G_E_and_interaction = reml_em(Kernel=Kernels_G_and_E, sig_estimate = np.array([0.33,0.33,0.33]), X=X, y=y, verbose=True, n_iter=200)[0]
variance_ratio_explained_by_G_E_and_interaction = (sigma_sqs_reml_Pheno_vs_G_E_and_interaction[0]+sigma_sqs_reml_Pheno_vs_G_E_and_interaction[1])/np.sum(sigma_sqs_reml_Pheno_vs_G_E_and_interaction)
print(sigma_sqs_reml_Pheno_vs_G_E_and_interaction)

#Partial regression trick
variance_ratio_explained_by_G_only = variance_ratio_explained_by_G_E_and_interaction - variance_ratio_explained_by_E_and_interaction
variance_ratio_explained_by_E_only = variance_ratio_explained_by_G_E_and_interaction - variance_ratio_explained_by_G_and_interaction
variance_ratio_explained_by_interaction_only_from_subs_G_only = variance_ratio_explained_by_G_and_interaction - variance_ratio_explained_by_G_only
variance_ratio_explained_by_interaction_only_from_subs_E_only = variance_ratio_explained_by_E_and_interaction - variance_ratio_explained_by_E_only

#print variance explained
print("Variance ratio of phenotype explained genotype only is {0}".format(variance_ratio_explained_by_G_only))
print("Variance ratio of phenotype explained expression only is {0}".format(variance_ratio_explained_by_E_only))
print("Variance ratio of phenotype explained genotype-expression interaction is {0} (as measured from R^2_G_and_i - R^2_G_only)".format(variance_ratio_explained_by_interaction_only_from_subs_G_only))
print("Variance ratio of phenotype explained genotype-expression interaction is {0} (as measured from R^2_E_and_i - R^2_E_only)".format(variance_ratio_explained_by_interaction_only_from_subs_E_only))

#save Phenotype variance explained in table
data = {'variance_ratio_explained_by_G_only' : [variance_ratio_explained_by_G_only],
        'variance_ratio_explained_by_E_only': [variance_ratio_explained_by_E_only],
        'variance_ratio_explained_by_interaction_only_from_subs_G_only' : [variance_ratio_explained_by_interaction_only_from_subs_G_only],
        'variance_ratio_explained_by_interaction_only_from_subs_E_only': [variance_ratio_explained_by_interaction_only_from_subs_E_only]}
df_out = pd.DataFrame(data)
df_out.to_csv("{0}/Table_phenotype_variance_explained.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)
