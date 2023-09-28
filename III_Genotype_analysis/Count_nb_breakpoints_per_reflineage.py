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

df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_lst_cells = pd.read_csv("{0}/lst_barcodes_with_expression_data.txt".format(workspace_path),sep="\t",header=None,dtype={ '0': str })
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]
data_wp_path = workspace_path+"/data"

#import ALL reference lineage genotypes
df_all_reference_lineage_genotypes = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,lst_label_chromosomes[0]),sep="\t",header=None)

for the_label_chr in lst_label_chromosomes[1:len(lst_label_chromosomes)]:
    df_to_add = pd.read_csv("{0}/{1}_spore_major.txt".format(yeast_project_wp_path,the_label_chr),sep="\t",header=None)
    df_to_add = df_to_add.reset_index(drop=True)
    df_all_reference_lineage_genotypes = pd.concat([df_all_reference_lineage_genotypes,df_to_add],axis=1)
mtx_all_reference_lineage_genotypes = df_all_reference_lineage_genotypes.to_numpy()

arr_nb_breakpoints_reflineages = np.zeros(np.shape(mtx_all_reference_lineage_genotypes)[0])

for i in np.arange(np.shape(mtx_all_reference_lineage_genotypes)[0]):
    current_reflineage_genotype = mtx_all_reference_lineage_genotypes[i,:].tolist()
    current_genotype = current_reflineage_genotype[0]
    old_genotype = current_genotype
    if (current_genotype >= 0.75):
        current_allele = "RM"
    elif (current_genotype <= 0.25):
        current_allele = "BY"
    if (old_genotype >= 0.75):
        old_allele = "RM"
    elif (old_genotype <= 0.25):
        old_allele = "BY"
    for current_genotype in current_reflineage_genotype:
        if (current_genotype >= 0.75):
            current_allele = "RM"
        elif (current_genotype <= 0.25):
            current_allele = "BY"
        if current_allele != old_allele:
            arr_nb_breakpoints_reflineages[i] = arr_nb_breakpoints_reflineages[i] + 1
        old_allele = current_allele
    print("Count of the number of breakpoints done for {0} reference lineages!".format(i+1))
data = {'id_lineage' : np.arange(np.shape(mtx_all_reference_lineage_genotypes)[0]),
        'nb_breakpoints' : arr_nb_breakpoints_reflineages.tolist()}
df_nb_breakpoints_reflineages = pd.DataFrame(data)
df_nb_breakpoints_reflineages["rank"] = df_nb_breakpoints_reflineages["nb_breakpoints"].rank()
df_nb_breakpoints_reflineages.to_csv("{0}/df_nb_breakpoints_reflineages.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)