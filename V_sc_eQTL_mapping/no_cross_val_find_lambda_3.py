# Here we're going to estimate lambda based on cross validation error across all folds instead of a single fold at a time.

import string
import numpy as np
import sys
import csv
import itertools
import time
import argparse
import os
import pandas as pd
cwd = os.getcwd()


from argparse import ArgumentParser, SUPPRESS
workspace_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[1]
yeast_project_wp_path = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[2]
cellranger_outs_folder = "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL" #sys.argv[3]

# Disable default help
parser = ArgumentParser(add_help=False)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

# Add back help
optional.add_argument(
    '-h',
    '--help',
    action='help',
    default=SUPPRESS,
    help='show this help message and exit'
)
required.add_argument('-dir', help='Parent directory', required=True)
required.add_argument('--fit', help='Phenotype data / Expression matrix in the case of sc_eQTL')
required.add_argument('--geneid', help='Gene id',type=int)
required.add_argument('-model', help='Find lambda according to the crossvalidation in the oCV (model = 2), or iCV (model = 1)', required=True, type=int)
optional.add_argument('--oCV', help='Outside cross-validation value (k = 0-9)', type=int, default=0)
optional.add_argument('--downsample', help='Number of segregants to downsample.', default=0, type=int)
optional.add_argument('-lambda_output', help='File to output the lambda value.')
optional.add_argument('--SE', help='Whether to output at lambda with 1 SE.', type=int, default=0)
optional.add_argument('--sporelist', help='Restrict searches to a list of spores.')
optional.add_argument('--unweighted', help='Only run the forward search on unweighted data.', default=0, type=int)
args = parser.parse_args()


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

# Read in the fitness data
mtx_gene_expression = pd.read_csv(args.fit,sep="\t",header=None).to_numpy()
#mtx_gene_expression = (mtx_gene_expression - mtx_gene_expression.mean(axis=0))/np.std(mtx_gene_expression,axis=0) #mtx_gene_expression = mtx_gene_expression/np.nansum(a=mtx_gene_expressio>

'''
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
'''
fitnesses_data = mtx_gene_expression[:,int(args.geneid)] #mean_E[:,int(args.geneid)] #np.loadtxt(args.fit)

counts = np.shape(mtx_gene_expression)[0]

del mtx_gene_expression

# Parse and see if it has standard errors

if(len(fitnesses_data.shape) != 2):
	# No errors found, assume all errors the same.
	if(len(fitnesses_data.shape) == 1):
		fitnesses_data = np.reshape(fitnesses_data,(-1,1))

	fitnesses = fitnesses_data[:,0]
	errors = np.ones(len(fitnesses_data))
else:
	fitnesses = fitnesses_data[:,0]
	errors = fitnesses_data[:,1]

seed = 100000
np.random.seed(seed) # Train set 1

sporelist = np.array(range(len(fitnesses)))
if(args.sporelist):
	sporelist = np.loadtxt(args.sporelist, dtype=int)

# First let's take care of the outside CV

if(args.downsample > 0 and args.downsample < len(sporelist)):
	#fitnesses = fitnesses[0:args.downsample]
	#errors = errors[0:args.downsample]
	sporelist = sporelist[0:args.downsample]


outside_CV = args.oCV # Goes from 0 to 9 # k = 10

if(outside_CV > 9 or outside_CV < 0):
	print("--oCV must be [0,9]")
	exit()

if(~np.isin(args.model , range(1,3))):
	print("--model must be [1,2]")
	exit()

perm = np.random.permutation(sporelist)

train_perm = perm.copy()
if(args.model != 2):
	train_perm = np.delete(train_perm, np.r_[outside_CV/10 * len(sporelist):(outside_CV + 1)/10 * len(sporelist)].astype(int),axis=0)

# else, we'll provide a train/test set and use the lassoCV function. Although, I'm not sure that lassoCV can handle our weird way of doing things... like having weighted train, unweighted test. Having an intercept during the train, but also adjusting one for the prediction.
# I think the best way is for us to code the subroutine ourselves.

train_set = np.take(fitnesses,train_perm) # If model = 2, then this is the whole set, if model = 1, then this is 90% of the data, and if model = 2 then this is 80% of the data.

train_errors = np.take(errors,train_perm)
train_phenotypes = train_set[~np.isnan(train_set)] # Is a numpy.ndarray
train_errors = train_errors[~np.isnan(train_set)]
train_num_usable_spores = len(train_phenotypes)
mean_train_phenotypes = np.mean(train_phenotypes)

TSS_2 = np.sum((train_phenotypes - mean_train_phenotypes)**2)

# Ok, let's parse the directory structure.
dir_struct = ""
max_range = ""
if(args.model == 2):
	dir_struct = "oCV_"
	max_range = 10
elif(args.model == 1):
	dir_struct = "iCV_"
	max_range = 9
else:
	print("-model must be 1 or 2.", file=sys.stderr)
	exit()

filehandle = sys.stderr
if(args.lambda_output):
	filehandle = open(args.lambda_output,"w")

'''
cross_val = []
header = []
counts = 0
lst_id_oCVs_with_variance = []
i_header = 0
for i in range(max_range):
	#print(str(args.dir) + "/" + str(dir_struct) + str(i) + "/" + str("cross_val.txt"))
	try:
		current_cross_val_body = pd.read_csv(str(args.dir) + "/" + str(dir_struct) + str(i) + "/cross_val.txt",sep="\t",header=None,skiprows=1).to_numpy()
		#current_cross_val_body = np.loadtxt(str(args.dir) + "/" + str(dir_struct) + str(i) + "/cross_val.txt", skiprows = 1)
	except Exception as e:
		#print("oCV {0} skipped!".format(i))
		#print(str(e))
		continue
	if (np.sum(~np.isfinite(current_cross_val_body[:,5]))>0):
		continue
	lst_id_oCVs_with_variance.append(i)
	cross_val.append(np.loadtxt(str(args.dir) + "/" + str(dir_struct) + str(i) + "/cross_val.txt", skiprows = 1))
	header.append(np.loadtxt(str(args.dir) + "/" + str(dir_struct) + str(i)+ "/cross_val.txt", max_rows = 1))
	counts = counts + header[i_header][0]
	i_header = i_header + 1
nb_oCVs_with_variance = len(lst_id_oCVs_with_variance)
if nb_oCVs_with_variance == 0:
	#print(args.geneid)
	exit()
# Ok, we have loaded all the data.
# Now, we'll start with a lambda, obtain the cross validation error. Then we'll change it, and maximize.

lambda_min = np.log(counts / nb_oCVs_with_variance)
lambda_try = lambda_min
# Now iterate
max_average_R2 = -10000
for iteration in range(200):
	average_R2 = 0
	for i in range(nb_oCVs_with_variance):
		likelihood = cross_val[i][:,1] + cross_val[i][:,0] * lambda_try

		average_R2 = average_R2 + cross_val[i][np.argmin(likelihood)][5]
	
	average_R2 = average_R2/nb_oCVs_with_variance
	#print(str(lambda_try) + "	" + str(average_R2))
	if(average_R2 >= max_average_R2):
		max_average_R2 = average_R2
		lambda_max = lambda_try * 1.05
		lambda_min = lambda_try / 1.05

	lambda_try = lambda_try * 1.05


# Found the brackets, now optimize within.
lambda_min = lambda_min / 2
if(lambda_min < np.log(counts/nb_oCVs_with_variance)):
	lambda_min = np.log(counts/nb_oCVs_with_variance)

phi = (1 + np.sqrt(5))/2
# There is a way to code this to prevent re-evaluating the functions. But that takes a bit more work to implement, so I won't do it unless needed.
#print(str(lambda_min) + "	" + str(lambda_max))
for iteration in range(100):
	lambda_left = lambda_min + (lambda_max - lambda_min)/(phi+1)
	lambda_right = lambda_max - (lambda_max - lambda_min)/(phi+1)

	# Functional evaluation on the left side
	R2_left = []
	for i in range(nb_oCVs_with_variance):
		likelihood = cross_val[i][:,1] + cross_val[i][:,0] * lambda_left
		#average_R2_left = average_R2_left + cross_val[i][np.argmin(likelihood)][5]
		R2_left.append(cross_val[i][np.argmin(likelihood)][5])
	
	average_R2_left = np.mean(R2_left)
	SE_R2_left = np.std(R2_left)/np.sqrt(len(R2_left))
	# Functional evaluation on the right side
	R2_right = []
	for i in range(nb_oCVs_with_variance):
		likelihood = cross_val[i][:,1] + cross_val[i][:,0] * lambda_right
		#average_R2_right = average_R2_right + cross_val[i][np.argmin(likelihood)][5]
		R2_right.append(cross_val[i][np.argmin(likelihood)][5])
	
	average_R2_right = np.mean(R2_right)
	SE_R2_right = np.std(R2_right)/np.sqrt(len(R2_right))

	if(average_R2_left > average_R2_right):
		# Maximum value is between lambda_min and lambda_right
		lambda_max = lambda_right
	else:
		lambda_min = lambda_left

	#print(str(lambda_min) + "	" + str(lambda_max) + "	" + str(average_R2_left) + "	" + str(average_R2_right))
'''

lambda_min =  np.log(counts)
lambda_max =  np.log(counts)
optimal_lambda = (lambda_max + lambda_min)/2
optimal_R2 = 0.0
optimal_SE = 0

'''
# Identify lambda where R2 crosses zero at R2 - optimal_SE
if(args.SE == 1):
	lambda_min = optimal_lambda
	lambda_max = lambda_min * 10


	for iteration in range(100):
		lambda_try = (lambda_max + lambda_min)/2
		# Functional evaluation on the left side
		R2 = []
		for i in range(nb_oCVs_with_variance):
			likelihood = cross_val[i][:,1] + cross_val[i][:,0] * lambda_try
			R2.append(cross_val[i][np.argmin(likelihood)][5])
	
		average_R2 = np.mean(R2) - (optimal_R2-optimal_SE)

		# Functional evaluation on the right side
	
		if(average_R2 < 0):
			# Maximum value is between lambda_min and lambda_right
			lambda_max = lambda_try
		else:
			lambda_min = lambda_try

	

	optimal_lambda = (lambda_max + lambda_min)/2 # This gets the 1SE lambda value for the forward search
'''

# Now parse the output file and retrieve the positions/betas for this. This is the unrefined positions at the cross validation lambda.

positions = []
effects = []
AICs = []

# Zeros
AICs.append(np.log(TSS_2/len(train_phenotypes)) * len(train_phenotypes))
positions.append(np.array([]))
effects.append(np.array([]))

with open(str(args.dir) + "/" + "output.txt",'r') as readfile:

	linecount = 0
	for line in readfile:
		line = line.rstrip()
		if(linecount % 4 == 0):
			AIC = float(line)
			AICs.append(AIC)

		if(linecount % 4 == 1):
			# positions
			pos = np.fromstring(line, sep="	",dtype=int)
			positions.append(pos)

		if(linecount %4 == 2):
			# Effects
			eff = np.fromstring(line, sep="	")
			effects.append(eff)

		linecount = linecount + 1

# Done parsing, now go through:

penalized_likelihood = []
for i in range(len(AICs)):
	penalized_likelihood.append(AICs[i] + optimal_lambda * len(positions[i]))
	#print(str(i) + "	" + str(penalized_likelihood[i]))

#print(penalized_likelihood)

print(str((lambda_max + lambda_min)/2) + "	" + str(optimal_R2) + "	" + str(optimal_SE) + "	" + str(len(positions[np.argmin(penalized_likelihood)])), file=filehandle)
print(*positions[np.argmin(penalized_likelihood)])
print(*effects[np.argmin(penalized_likelihood)])
