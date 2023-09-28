import string
import numpy as np
from scipy import linalg
import sys
import csv
import itertools
import time
import random
import argparse
import os
import sys
import time
from contextlib import closing
import scipy.io
from scipy.stats import mannwhitneyu, linregress, rankdata, skew
import numpy.linalg as LA
import pandas as pd
cwd = os.getcwd()


#import main workspace absolute path
the_gene_id = int(sys.argv[1])
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[2]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[3]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast" #sys.argv[4]
sc_eqtl_workspace = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL"

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

chrom_startpoints = [0, 996, 4732, 5291, 9327, 11187, 12476, 16408, 18047, 20126, 23101, 26341, 30652, 33598, 35398, 39688]
chrom_endpoints = [994, 4730, 5289, 9325, 11185, 12474, 16406, 18045, 20124, 23099, 26339, 30650, 33596, 35396, 39686, 41608]
num_SNPs = [995, 3735, 558, 4035, 1859, 1288, 3931, 1638, 2078, 2974, 3239, 4310, 2945, 1799, 4289, 1921]

#open list of eQTLs for current gene
with open("{0}/gene_id_{1}/refined_merged.txt".format(sc_eqtl_workspace,the_gene_id)) as f:
    firstline = f.readline().rstrip()
    secondline = f.readline().rstrip()
lst_eqtls_current_gene = firstline.split()
lst_eqtls_current_gene = [int(x) for x in lst_eqtls_current_gene]
print("lst_eqtls_current_gene:")
print(lst_eqtls_current_gene)
lst_betas = secondline.split()
lst_betas = [float(x) for x in lst_betas]
print("lst_betas:")
print(lst_betas)

# Open all the genotype files
num_lines_genotypes = []
chr_to_scan = []
lst_indexes_snps_in_org_SNP_pos_table = []
start = time.perf_counter()

for ind_pos_QTL_scan in np.arange(len(chrom_startpoints)):
    lst_indexes_snps_in_org_SNP_pos_table.extend(np.arange(chrom_startpoints[ind_pos_QTL_scan],chrom_endpoints[ind_pos_QTL_scan]+1,1).tolist())

id_column_QTLs = (np.array(lst_indexes_snps_in_org_SNP_pos_table)[lst_eqtls_current_gene,]).tolist()
print("id_column_QTLs:")
print(id_column_QTLs)
df_out = df_pos_snps.iloc[id_column_QTLs,:]
df_out['betas'] = lst_betas
df_out['gene_id'] = [the_gene_id]*(np.shape(df_out)[0])
df_out.to_csv("{0}/df_pos_snps_eQTLs_gene_{1}.csv".format(sc_eqtl_workspace,the_gene_id), sep='\t',na_rep="NA",header=False,index=False)

