# Here we're going to estimate lambda based on cross validation error across all folds instead of a single fold at a time.

import string
import numpy as np
import sys
import csv
import itertools
import time
import argparse
import os
cwd = os.getcwd()


from argparse import ArgumentParser, SUPPRESS
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
required.add_argument('-model', help='Whether to find lambda in oCV or in iCV directories. 0 = iCV, 1 = oCV', required=True, type=int)
optional.add_argument('-lambda_output', help='File to output the lambda value.')
optional.add_argument('--SE', help='Whether to output at lambda with 1 SE.', type=int, default=0)

args = parser.parse_args()

# Ok, let's parse the directory structure.
dir_struct = ""
max_range = ""
if(args.model == 1):
	dir_struct = "oCV_"
	max_range = 10
elif(args.model == 0):
	dir_struct = "iCV_"
	max_range = 9
else:
	print("-model must be 0 or 1.", file=sys.stderr)
	exit()

filehandle = sys.stderr
if(args.lambda_output):
	filehandle = open(args.lambda_output,"w")


cross_val = []
header = []
counts = 0
for i in range(max_range):
	#print(str(args.dir) + "/" + str(dir_struct) + str(i) + "/" + str("cross_val.txt"))
	cross_val.append(np.loadtxt(str(args.dir) + "/" + str(dir_struct) + str(i) + "/" + str("cross_val_epistasis.txt"), skiprows = 1))
	header.append(np.loadtxt(str(args.dir) + "/" + str(dir_struct) + str(i)+ "/" + str("cross_val_epistasis.txt"), max_rows = 1))
	counts = counts + header[i][0]

# Ok, we have loaded all the data.
# Now, we'll start with a lambda, obtain the cross validation error. Then we'll change it, and maximize.

lambda_min = np.log(counts / max_range)
#print(lambda_min)
lambda_try = lambda_min
# Now iterate
max_average_R2 = 0
for iteration in range(20):
	average_R2 = 0
	for i in range(max_range):
		likelihood = cross_val[i][:,1] + cross_val[i][:,0] * lambda_try

		average_R2 = average_R2 + cross_val[i][np.argmin(likelihood)][5]
	
	average_R2 = average_R2/max_range
	#print(str(lambda_try) + "	" + str(average_R2))
	if(average_R2 > max_average_R2):
		max_average_R2 = average_R2
	else:
		lambda_max = lambda_try
		break
	lambda_min = lambda_try
	lambda_try = lambda_try * 2


# Found the brackets, now optimize within.
lambda_min = lambda_min / 2
if(lambda_min < np.log(counts/max_range)):
	lambda_min = np.log(counts/max_range)

phi = (1 + np.sqrt(5))/2
# There is a way to code this to prevent re-evaluating the functions. But that takes a bit more work to implement, so I won't do it unless needed.
#print(str(lambda_min) + "	" + str(lambda_max))
for iteration in range(300):
	lambda_left = lambda_min + (lambda_max - lambda_min)/(phi+1)
	lambda_right = lambda_max - (lambda_max - lambda_min)/(phi+1)

	# Functional evaluation on the left side
	R2_left = []
	for i in range(max_range):
		likelihood = cross_val[i][:,1] + cross_val[i][:,0] * lambda_left
		#average_R2_left = average_R2_left + cross_val[i][np.argmin(likelihood)][5]
		R2_left.append(cross_val[i][np.argmin(likelihood)][5])
	
	average_R2_left = np.mean(R2_left)
	SE_R2_left = np.std(R2_left)/np.sqrt(len(R2_left))
	# Functional evaluation on the right side
	R2_right = []
	for i in range(max_range):
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


optimal_lambda = (lambda_max + lambda_min)/2 
optimal_R2 = (average_R2_left + average_R2_right)/2
optimal_SE = (SE_R2_left + SE_R2_right)/2

# Identify lambda where R2 crosses zero at R2 - optimal_SE
if(args.SE == 1):
	lambda_min = optimal_lambda
	lambda_max = lambda_min * 10


	for iteration in range(100):
		lambda_try = (lambda_max + lambda_min)/2
		# Functional evaluation on the left side
		R2 = []
		for i in range(max_range):
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

# Now parse the output file and retrieve the positions/betas for this. This is the unrefined positions at the cross validation lambda.

positions = []
effects = []
AICs = []
with open(str(args.dir) + "/" + "epistasis.txt",'r') as readfile:

	linecount = 0
	for line in readfile:
		line = line.rstrip()
		if(line == "Done"):
			continue
		if(linecount % 4 == 0):
			AIC = float(line)
			AICs.append(AIC)
		
		if(linecount % 4 == 1):
			# positions
			pos = line.split("	")
			#print(len(pos))
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

print(str((lambda_max + lambda_min)/2) + "	" + str(optimal_R2) + "	" + str(optimal_SE) + "	" + str(len(positions[np.argmin(penalized_likelihood)])), file=filehandle)


print(*positions[np.argmin(penalized_likelihood)],sep="\t")
print(*effects[np.argmin(penalized_likelihood)])

