import string
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import itertools
import math

#Define genome parameters
chrom_lengths = [230218,813184,316620,1531933,576874,270161,1090940,562643,439888,745751,666816,1078177,924431,784333,109291,948066]
num_chroms = 16

# class SNP:
#
# 	def __init__(self,chromosome,chrom_loc,parent=np.nan):
# 		self.chrom = chromosome
# 		self.chrom_loc = chrom_loc
# 		self.value = parent_to_int(parent)
#
#
# class SNP_map:
#
# 	def __init__(self,name,genomelist=None):
# 		self.name = name
# 		if genomelist:
# 			self.genome = genome_to_chroms(genomelist)
# 		else:
# 			self.genome = [[0,i] for i in chrom_lengths]
#
#
# 	def print_SNPs(self):
# 		for i in range(len(self.genome)):
# 			print("Chrom "+str(i+1)+": ",self.genome[i])
#
# 	def print_SNPs_chrom(self,chrom):
# 		print("Chrom "+str(chrom)+": ",self.genome[chrom-1])
#
#
# 	def insert_SNP(self,this_SNP):
# 		self.genome[this_SNP.chrom-1].append(this_SNP.chrom_loc)
# 		self.genome[this_SNP.chrom-1].sort()
#
# 	def get_num_SNPs(self):
# 		num_SNPs = []
# 		for i in range(len(self.genome)):
# 			num_SNPs.append(len(self.genome[i]))
# 		return(num_SNPs)
#
# 	def get_total_SNPs(self):
# 		num_SNPs = 0
# 		for i in range(len(self.genome)):
# 			num_SNPs += len(self.genome[i])
# 		return(num_SNPs)
#
# 	def get_SNP_index(self,this_SNP):
# 		index = self.genome[this_SNP.chrom-1].index(this_SNP.chrom_loc)
# 		return(index-2)
#
# 	def get_SNP_dist_between(self,chrom,index1,index2):
# 		dist_bp = math.fabs(self.genome[chrom][index2+1]-self.genome[chrom][index1+1])
# 		return(dist_bp)
#
# 	def get_SNPs_in_interval(self,chrom,center,dist):
#
# 		left = center-1
# 		while math.fabs(self.genome[chrom][center+1]-self.genome[chrom][left+1]) < dist:
# 			if left == 0: break
# 			left -= 1
# 		right = center+1
# 		while math.fabs(self.genome[chrom][center+1]-self.genome[chrom][right+1]) < dist:
# 			if right == len(self.genome[chrom])-2: break
# 			right += 1
#
# 		if center == 0:
# 			left = center
# 		elif center == len(self.genome[chrom])-2:
# 			right = self.genome[chrom][-2]
# 		return(left,right)
#
# 	def get_chrom_endpoints(self):
# 		num_SNPs = []
# 		for i in range(len(self.genome)):
# 			num_SNPs.append(len(self.genome[i])-2)
#
# 		chrom_endpoints = []
# 		loc = -1
# 		for x in range(num_chroms-1):
# 			chrom_endpoints.append(loc+num_SNPs[x]+1)
# 			loc = loc+num_SNPs[x]+1
# 		return(chrom_endpoints)

# class Spore:
#
# 	def __init__(self,cross_map,haplolist=None):
# 		chrom_SNPs = cross_map.get_num_SNPs()
#
# 		self.map = cross_map
# 		self.fitness = np.nan
# 		self.address = (np.nan,np.nan,np.nan,np.nan,np.nan)
# 		self.barcode = ('X','X')
#
# 		if haplolist:
# 			self.haplotype = genome_to_chroms(haplolist)
# 		else:
# 			self.haplotype = []
# 			for i in range(len(chrom_SNPs)):
# 				self.haplotype.append([])
#
# 	def append_SNP(self,this_SNP):
# 		self.haplotype[this_SNP.chrom-1].append(this_SNP.value)
#
#
# 	def insert_SNP(self,this_SNP):
# 		self.haplotype[this_SNP.chrom-1][self.map.get_SNP_index(this_SNP)] = this_SNP.value
#
# 	def print_haplotype(self):
# 		for i in range(len(self.haplotype)):
# 			print("Chrom "+str(i+1)+": ",self.haplotype[i])
#
# 	def print_chrom(self,chrom):
# 		print("Chrom "+str(chrom+1)+": ",self.haplotype[chrom])
#
# 	def print_address(self):
# 		address = [str(x) for x in self.address]
# 		print("Cross "+address[0]+" Batch "+address[1]+" Set "+address[2]+" Plate "+address[3]+" Well "+address[4])
#
# 	def print_barcode(self):
# 		print(self.barcode)
#
# 	def write_to_file(self):
# 		line = [x for x in self.address]
# 		line.append(self.barcode[0])
# 		line.append(self.barcode[1])
# 		line.append(self.fitness)
# 		line.append(chroms_to_genome(self.haplotype))
# 		return(line)
# #
# class QTL_map:
#
# 	def __init__(self,cross_map,qtl_list=None):
# 		chrom_SNPs = cross_map.get_num_SNPs()
#
# 		self.map = cross_map
#
# 		if len(qtl_list) > 0:
# 			self.genome = genome_to_chroms(qtl_list)
# 		else:
# 			self.genome = []
# 			for i in range(len(chrom_SNPs)):
# 				self.genome.append([])
# def get_fitness(QTL_map,spore_haplotype):
# 	s = np.full(len(spore_haplotype),0.0)
# 	for chrom in range(len(s)):
# 		spore_chrom = np.array(spore_haplotype[chrom])
# 		QTL_chrom = np.array(QTL_map.genome[chrom])
# 		s[chrom] = np.sum(QTL_chrom*spore_chrom)
#
# 	return np.sum(s)
		
	
def parent_to_int(parent):
	if parent == '0' or parent == '1': return(int(parent))
	else: return(parent)

def chroms_to_genome(genome_chroms):
	genome = []
	for i in range(len(genome_chroms)):
		for j in range(len(genome_chroms[i])):
			genome.append(genome_chroms[i][j])
		genome.append(2.0)
	return(genome[:-1])
	
def genome_to_chroms(genome):
	chroms = split(genome,[2.0])
	return(chroms)
	
def reads_to_chroms(genome):
	chroms = split(genome,[-1.0,-2.0])
	return(chroms)

def chroms_to_reads(genome_chroms):
	genome = []
	for i in range(len(genome_chroms)):
		for j in range(len(genome_chroms[i])):
			genome.append(genome_chroms[i][j])
		genome.append(-1.0)
	return(genome[:-1])

def genome_str_to_int(genome):
	for i in range(len(genome)):
		if genome[i] == 'nan': genome[i] = np.nan
		elif genome[i] == '2.0': genome[i] = 2.0
		else: genome[i] = int(genome[i])
	return(genome)

def genome_str_to_float(genome):
	for i in range(len(genome)):
		if genome[i] == 'nan': genome[i] = np.nan
		elif genome[i] == '2.0': genome[i] = 2.0
		else: genome[i] = float(genome[i])
	return(genome)

def get_chrom_startpoints(genome):
	# indices of the first SNP on each chromosome 
	split_values = [i for i in range(len(genome)) if genome[i] == 2.0]
	chrom_startpoints = [i+1 for i in split_values]
	chrom_startpoints.insert(0,0)
	return(chrom_startpoints)
	
def get_chrom_endpoints(genome):
	# indices of the last SNP on each chromosome
	split_values = [i for i in range(len(genome)) if genome[i] == 2.0]
	chrom_endpoints = [i-1 for i in split_values]
	chrom_endpoints.append(len(genome)-1)
	return(chrom_endpoints)


def split(iterable,splitval):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitval) if not k]
	

def haldane(d):
	# distance d is in basepairs
	# 1 cM = 2 kb
	return 0.5*(1-np.exp(-2*d*(0.01/2000)))
