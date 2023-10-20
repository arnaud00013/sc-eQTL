#Script for estimating HMM error rates parameters a multi-barcode bam file
#import libraries
from pathlib import Path
import pysam
import sys
import time
from multiprocessing import Pool, TimeoutError, Array
from contextlib import closing
import os
import numpy as np
import pandas as pd

# load samtools in environment
os.system("module load samtools/1.12")

#import reference data and script argument
workspace_path = sys.argv[1]
the_input_bam = sys.argv[2]
the_barcodes_file = sys.argv[3]
cutoff_nb_mismatches = int(sys.argv[4])
cutoff_coverage = int(sys.argv[5])
nb_cpus = int(sys.argv[6])
df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(workspace_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]
df_pos_snps.index = df_pos_snps["the_key"].tolist()

#List of barcodes and chromosomes
df_lst_cells = pd.read_csv(the_barcodes_file,sep="\t",header=None,dtype={ '0': str })
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]

BY_file = pysam.FastaFile("{0}/BY4742_fixed.fa".format(workspace_path))
RM_file = pysam.FastaFile("{0}/RM11a.fa".format(workspace_path)) #RM_file.fetch(contig)[the_pos]
dict_chrs_refseq = {}
for current_chr in lst_label_chromosomes:
    dict_chrs_refseq[current_chr] = RM_file.fetch(current_chr)
    df_pos_snps_current_chr = df_pos_snps[df_pos_snps.chromosome==current_chr]
    #for i in np.arange(np.shape(df_pos_snps_current_chr)[0]):
        #assert(df_pos_snps_current_chr.at["{0}_{1}".format(current_chr,df_pos_snps_current_chr.position.tolist()[i]),"Allele"] == dict_chrs_refseq[current_chr][df_pos_snps_current_chr.position.tolist()[i]-1])

#create a dataframe with the number of reads per chromosome
try:
    df_nb_reads_per_chromosome = pd.read_csv("{0}/nb_reads_per_chr_NoFastp_Yeast.txt".format(workspace_path),sep="\t",dtype={ '0': str, '1': int })
except:
    os.system("samtools idxstats {0}/data/{1} | cut -f 1,3 | grep chr > {0}/nb_reads_per_chr_NoFastp_Yeast.txt".format(workspace_path,the_input_bam))
    df_nb_reads_per_chromosome = pd.read_csv("{0}/nb_reads_per_chr_NoFastp_Yeast.txt".format(workspace_path),sep="\t",dtype={ '0': str, '1': int })

df_nb_reads_per_chromosome.columns = ["chromosome","nb_reads"]
df_nb_reads_per_chromosome.reindex(df_nb_reads_per_chromosome.chromosome.tolist())
print("df_nb_reads_per_chromosome:")
print(df_nb_reads_per_chromosome)

#subsample of the mapping file (estimate the error rates from a subsample of a million reads)
try:
    #make sure that the subsampled bamfile is created
    bamfile = pysam.AlignmentFile("{0}/data/subsample_mapping.bam".format(workspace_path))
    bamfile.close()
    #make sure that the subsampled bamfile copy is created
    bamfile_pileup = pysam.AlignmentFile("{0}/data/cp_subsample_mapping.bam".format(workspace_path))
    bamfile_pileup.close()
except:
    #create the subsampled bam file
    os.system("samtools view -s {0} -b {1}/data/{2} > {1}/data/subsample_mapping.bam".format(1234+(1000000/np.sum(df_nb_reads_per_chromosome.nb_reads.tolist())),workspace_path,the_input_bam))
    #sort and index the subsampled bam file
    os.system("samtools sort {0}/data/subsample_mapping.bam -o {0}/data/tmp && mv {0}/data/tmp {0}/data/subsample_mapping.bam".format(workspace_path))
    os.system("samtools index {0}/data/subsample_mapping.bam".format(workspace_path))
    os.system("cp {0}/data/subsample_mapping.bam {0}/data/cp_subsample_mapping.bam".format(workspace_path))
    os.system("samtools index {0}/data/cp_subsample_mapping.bam".format(workspace_path))

print("proportion of reads selected for downsampling = {0}".format((1000000/np.sum(df_nb_reads_per_chromosome.nb_reads.tolist()))))

'''
#import lineage assignment dataframe
df_lineage_assignment_cells = pd.read_csv("{0}/df_best_match_corrected_cell_gen_vs_batch1.csv".format(workspace_path),sep="\t",dtype={ '0': int, '1': np.float32, '2': np.float32 })
df_lineage_assignment_cells.columns = ["best_match", "min_dist","pvalue"]
df_lineage_assignment_cells.index = lst_cells
'''

#strings hamming distance (nb mismatch)
def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

#Count the number of mismatch if there are gaps
def get_nb_mismatches_in_nonsnp_and_snp_sites(the_cigar_tuple,the_readseq,the_refseq,the_ref_positions,the_lst_snp_pos_in_chr_refseq,the_lst_snp_pos_well_enough_covered):
    #initialize variables
    lst_pos_in_aln = []
    i_read = 0
    i_refseq = 0
    read_seq_in_aln = ""
    refseq_in_aln = ""
    #reconstruction of the alignment WITH GAPS
    for current_tuple in the_cigar_tuple:
        if (current_tuple[0]) in [0,7,8]: #matchORmismatch;softClipping;match;mismatch
            read_seq_in_aln = read_seq_in_aln + the_readseq[i_read:(i_read+(current_tuple[1]))]
            i_read = i_read + (current_tuple[1])
            refseq_in_aln = refseq_in_aln + the_refseq[i_refseq:(i_refseq+(current_tuple[1]))]
            lst_pos_in_aln.extend(the_ref_positions[i_refseq:(i_refseq+(current_tuple[1]))])
            i_refseq = i_refseq+(current_tuple[1])
        elif (current_tuple[0]) == 1: #Insertion
            read_seq_in_aln = read_seq_in_aln + the_readseq[i_read:(i_read+(current_tuple[1]))]
            i_read = i_read+(current_tuple[1])
            refseq_in_aln = refseq_in_aln + ("-"*(current_tuple[1]))
            lst_pos_in_aln.extend([float("nan")]*(current_tuple[1]))
        elif (current_tuple[0]) == 2: #Deletion
            #insert gaps in read
            read_seq_in_aln = read_seq_in_aln + ("-"*(current_tuple[1]))
            refseq_in_aln = refseq_in_aln + the_refseq[i_refseq:(i_refseq+(current_tuple[1]))]
            lst_pos_in_aln.extend(the_ref_positions[i_refseq:(i_refseq+(current_tuple[1]))])
            i_refseq = i_refseq+c(current_tuple[1])
        else:
            continue
    #Index in alignment for non-SNP positions and SNP positions WITH ENOUGH COVERAGE
    the_non_snp_pos_in_aln = [indxx for indxx in np.arange(len(lst_pos_in_aln)) if (not np.isnan(lst_pos_in_aln[indxx])) and (not lst_pos_in_aln[indxx] in the_lst_snp_pos_in_chr_refseq)]
    the_snp_pos_in_aln = [indxx for indxx in np.arange(len(lst_pos_in_aln)) if (not np.isnan(lst_pos_in_aln[indxx])) and (lst_pos_in_aln[indxx] in the_lst_snp_pos_well_enough_covered)]

    #Number of mismatches in non-SNP positions
    if (len(the_non_snp_pos_in_aln)==0):
        nb_mismatches_in_non_snp_positions = 0
    else:
        the_indices = [kk for kk in the_non_snp_pos_in_aln if not read_seq_in_aln[kk] in ["-","N"]]
        readseq_aln_seq_in_non_snp_pos = "".join([read_seq_in_aln[kk] for kk in the_indices])
        refseq_aln_seq_in_non_snp_pos = "".join([refseq_in_aln[kk] for kk in the_indices])
        nb_mismatches_in_non_snp_positions = hamming_distance(readseq_aln_seq_in_non_snp_pos,refseq_aln_seq_in_non_snp_pos)
    #Number of mismatches in SNP positions with enough coverage
    if (len(the_snp_pos_in_aln)==0):
        nb_mismatches_in_snp_positions_with_enough_cov = 0
    else:
        the_indices = [kk for kk in the_snp_pos_in_aln if not read_seq_in_aln[kk] in ["-","N"]]
        readseq_aln_seq_in_snp_pos = "".join([read_seq_in_aln[kk] for kk in the_indices])
        refseq_aln_seq_in_snp_pos = "".join([refseq_in_aln[kk] for kk in the_indices])
        nb_mismatches_in_snp_positions_with_enough_cov = hamming_distance(readseq_aln_seq_in_snp_pos,refseq_aln_seq_in_snp_pos)
    if (len(the_non_snp_pos_in_aln)!=0) and (len(the_snp_pos_in_aln)!=0):
        if nb_mismatches_in_non_snp_positions/(len(readseq_aln_seq_in_non_snp_pos)-readseq_aln_seq_in_non_snp_pos.count("N"))>0.1:
            print("{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10}".format(the_cigar_tuple,the_readseq,the_refseq,the_non_snp_pos_in_aln,the_snp_pos_in_aln,readseq_aln_seq_in_non_snp_pos,refseq_aln_seq_in_non_snp_pos,nb_mismatches_in_non_snp_positions,(len(readseq_aln_seq_in_non_snp_pos)-readseq_aln_seq_in_non_snp_pos.count("N")),the_ref_positions,lst_pos_in_aln))
    return (nb_mismatches_in_non_snp_positions,nb_mismatches_in_snp_positions_with_enough_cov)

#initialize output array
init_mtx_output = np.zeros((len(lst_label_chromosomes),4))
mtx_output = Array('d', len(lst_label_chromosomes)*4)
np_mtx_output = np.frombuffer(mtx_output.get_obj()).reshape((len(lst_label_chromosomes),4))
np.copyto(np_mtx_output, init_mtx_output)

#create a log file for per-read error rates
os.system("rm -f {0}/LOG_reads_err_rates_estimation.txt".format(workspace_path))
os.system("touch {0}/LOG_reads_err_rates_estimation.txt".format(workspace_path))

#function to parse multicell bam reads from a specific chromosome
def parse_chr_reads(index_chr):
    global mtx_output

    np_mtx_output = np.frombuffer(mtx_output.get_obj()).reshape((len(lst_label_chromosomes),4))

    #open alignment file
    bamfile_pileup = pysam.AlignmentFile("{0}/data/cp_subsample_mapping.bam".format(workspace_path), "rb")
    bamfile = pysam.AlignmentFile("{0}/data/subsample_mapping.bam".format(workspace_path), "rb")
    the_chr = "chr%02d"%(index_chr+1,)
    df_pos_snps_current_chr = df_pos_snps[df_pos_snps.chromosome==current_chr]
    i = 1
    for read in bamfile.fetch(region=the_chr):
        #reference sequence aligned to read
        ref_positions = read.get_reference_positions()
        refseq = "".join([dict_chrs_refseq[the_chr][zerobased_pos] for zerobased_pos in ref_positions])
        #list of snp positions in aligned positions
        lst_snp_pos_in_chr_refseq = np.intersect1d(ref_positions,df_pos_snps_current_chr.position.tolist()).tolist() #[x for x in ref_positions if x in df_pos_snps_current_chr.position.tolist()]
        #list of non-snp positions in aligned positions
        lst_index_non_snp_pos_in_chr_refseq = [x for x in np.arange(len(ref_positions)) if ref_positions[x] not in lst_snp_pos_in_chr_refseq]
        #original read full sequence
        org_readseq = read.query_sequence
        #aligned read
        read_aligned_seq = read.query_alignment_sequence
        #current reads cigar tuples
        current_reads_cigar_tuples = read.cigartuples
        #current reads cigar string
        current_reads_cigar_str = read.cigarstring
        #aligned read sequence in non-SNP position
        read_aligned_seq_not_in_snp_sites = "".join([read_aligned_seq[x] for x in lst_index_non_snp_pos_in_chr_refseq])
        #aligned reference sequence in non-SNP position
        refseq_aligned_seq_not_in_snp_sites = "".join([refseq[x] for x in lst_index_non_snp_pos_in_chr_refseq])
        #Consider the possibility of gaps in the read alignment
        if  current_reads_cigar_str.find('D') == -1 and current_reads_cigar_str.find('I') == -1: #No Gaps
            #count the number of reads where the number of snp positions with a minor SNV (not likely to be the cell allele at well covered sites) is >= cutoff_nb_mismatches
            lst_snp_pos_well_enough_covered = []
            nb_snp_sites_well_covered = 0
            nb_snp_sites_mismatch = 0
            coverage_current_position = 0
            if len(lst_snp_pos_in_chr_refseq) > 0:
                for current_snp_pos in lst_snp_pos_in_chr_refseq:
                    current_allele = read_aligned_seq[ref_positions.index(current_snp_pos)]
                    lst_alleles_mapping_at_snp_pos = []
                    for pileupcolumn in bamfile_pileup.pileup(contig=the_chr,start=current_snp_pos-1,stop=current_snp_pos,truncate=1):
                        coverage_current_position = pileupcolumn.n
                        for pileupread in pileupcolumn.pileups:
                            if not pileupread.is_del and not pileupread.is_refskip:
                                lst_alleles_mapping_at_snp_pos.append(pileupread.alignment.query_sequence[pileupread.query_position])
                    if coverage_current_position >= cutoff_coverage:
                        lst_snp_pos_well_enough_covered.append(current_snp_pos)
                        nb_snp_sites_well_covered = nb_snp_sites_well_covered + 1
                        freq_read_allele = np.sum(np.array(lst_alleles_mapping_at_snp_pos)==current_allele) / len(lst_alleles_mapping_at_snp_pos)
                        if freq_read_allele > 0 and freq_read_allele < 0.5:
                            nb_snp_sites_mismatch = nb_snp_sites_mismatch + 1
                        if freq_read_allele == 0: #possible deletion or refskip site
                            #print("Read allele not in pileup for Position: {0}_{1}; Read allele: {2}".format(the_chr,current_snp_pos,current_allele))
                            #print("List of pileup alleles")
                            #print(lst_alleles_mapping_at_snp_pos)
                            #sys.exit('Logical error: read allele has frequency 0 in pileup!')
                            continue
                if nb_snp_sites_well_covered >= cutoff_nb_mismatches:
                    np_mtx_output[index_chr,3] = np_mtx_output[index_chr,3] + 1
                    if nb_snp_sites_mismatch >= cutoff_nb_mismatches:
                        np_mtx_output[index_chr,2] = np_mtx_output[index_chr,2] + 1
            #Do not consider ambiguous read bases
            the_indices = [kk for kk in np.arange(len(read_aligned_seq_not_in_snp_sites)) if read_aligned_seq_not_in_snp_sites[kk] != "N"]
            read_aligned_seq_not_in_snp_sites = "".join([read_aligned_seq_not_in_snp_sites[kk] for kk in the_indices])
            refseq_aligned_seq_not_in_snp_sites = "".join([refseq_aligned_seq_not_in_snp_sites[kk] for kk in the_indices])
            #compute the hamming distance over the aligned bases
            the_hamming_dist = hamming_distance(read_aligned_seq_not_in_snp_sites,refseq_aligned_seq_not_in_snp_sites)
        else: #Gap in read alignment
            #Ignore reads with wrong mapping (incosistent length)
            #initialize variables
            nb_gaps_in_refseq = 0
            nb_gaps_in_read = 0
            #count number of inserted and deleted bases in read
            for current_tuple in current_reads_cigar_tuples:
                if current_tuple[0] == 1:
                    nb_gaps_in_refseq = nb_gaps_in_refseq + current_tuple[1]
                elif current_tuple[0] == 2:
                    nb_gaps_in_read = nb_gaps_in_read + current_tuple[1]
                else:
                    continue
            #ignore the deletion if it is an inconsistent single deletion of 1+ bases
            if (len(read_aligned_seq) + nb_gaps_in_read) != (len(refseq) + nb_gaps_in_refseq):
                #print("Ignoring the deletion because it contains an inconsistent single deletion of 1+ bases! \'{0};{1};{2};{3};{4};{5};{6};{7};{8}\'".format(read.query_name,the_chr,np.min(ref_positions),np.max(ref_positions),read_aligned_seq_not_in_snp_sites,refseq_aligned_seq_not_in_snp_sites,read.cigarstring,(len(read_aligned_seq) + nb_gaps_in_read),(len(refseq) + nb_gaps_in_refseq)))
                continue
            lst_snp_pos_well_enough_covered = []
            nb_snp_sites_well_covered = 0
            nb_snp_sites_mismatch = 0
            coverage_current_position = 0
            if len(lst_snp_pos_in_chr_refseq) > 0:
                for current_snp_pos in lst_snp_pos_in_chr_refseq:
                    current_allele = read_aligned_seq[ref_positions.index(current_snp_pos)]
                    lst_alleles_mapping_at_snp_pos = []
                    for pileupcolumn in bamfile_pileup.pileup(contig=the_chr,start=current_snp_pos-1,stop=current_snp_pos,truncate=1):
                        coverage_current_position = pileupcolumn.n
                        for pileupread in pileupcolumn.pileups:
                            if not pileupread.is_del and not pileupread.is_refskip:
                                lst_alleles_mapping_at_snp_pos.append(pileupread.alignment.query_sequence[pileupread.query_position])
                    if coverage_current_position >= cutoff_coverage:
                        lst_snp_pos_well_enough_covered.append(current_snp_pos)
                        nb_snp_sites_well_covered = nb_snp_sites_well_covered + 1
            current_read_aln_analysis = get_nb_mismatches_in_nonsnp_and_snp_sites(current_reads_cigar_tuples,read_aligned_seq,refseq,ref_positions,lst_snp_pos_in_chr_refseq,lst_snp_pos_well_enough_covered)
            the_hamming_dist = current_read_aln_analysis[0]
            nb_snp_sites_mismatch = current_read_aln_analysis[1]
            #update index swapping counts
            if len(lst_snp_pos_in_chr_refseq) > 0:
                if nb_snp_sites_well_covered >= cutoff_nb_mismatches:
                    np_mtx_output[index_chr,3] = np_mtx_output[index_chr,3] + 1
                    if nb_snp_sites_mismatch >= cutoff_nb_mismatches:
                        np_mtx_output[index_chr,2] = np_mtx_output[index_chr,2] + 1

        #update seqerror counts
        np_mtx_output[index_chr,0] = np_mtx_output[index_chr,0] + the_hamming_dist
        np_mtx_output[index_chr,1] = np_mtx_output[index_chr,1] + len(read_aligned_seq_not_in_snp_sites) - (read_aligned_seq_not_in_snp_sites.count("N"))
        if i%1000==0:
            print("read seq error rate is {0}".format(the_hamming_dist/(len(read_aligned_seq_not_in_snp_sites)-(read_aligned_seq_not_in_snp_sites.count("N")))))
            print("{0} reads parsed for chromosome {1}".format(i,the_chr))
        #save read sequencing error rate to LOG file for contamination analysis
        os.system("echo \'{0};{1};{2};{3};{4};{5};{6};{7}\' >> {8}/LOG_reads_err_rates_estimation.txt".format(read.query_name,the_chr,np.min(ref_positions),np.max(ref_positions),the_hamming_dist/(len(read_aligned_seq_not_in_snp_sites)-(read_aligned_seq_not_in_snp_sites.count("N"))),read_aligned_seq_not_in_snp_sites,refseq_aligned_seq_not_in_snp_sites,read.cigarstring,workspace_path))
        i = i + 1
    bamfile.close()
    bamfile_pileup.close()

# start pool of parallel worker processes
with closing(Pool(processes=nb_cpus)) as pool:
    pool.map(parse_chr_reads, np.arange(len(lst_label_chromosomes)))
    pool.terminate()

#Save data
data = {'nb_bp_seqerr_only':[np.sum(np_mtx_output[:,0])],
        'nb_tot_bp_reads_for_perr':[np.sum(np_mtx_output[:,1])],
        'P_err':[np.sum(np_mtx_output[:,0])/np.sum(np_mtx_output[:,1])],
        'nb_reads_index_swapping':[np.sum(np_mtx_output[:,2])],
        'nb_tot_reads_with_enough_well_covered_snp_sites':[np.sum(np_mtx_output[:,3])],
        'P_err_2':[np.sum(np_mtx_output[:,2])/np.sum(np_mtx_output[:,3])]}

# Create DataFrame
df_out = pd.DataFrame(data)
df_out.to_csv("{0}/Estimated_rates_Perrs_HMM.csv".format(workspace_path), sep='\t',na_rep="NA",header=True,index=False)

print("Perr = {0}".format(np.sum(np_mtx_output[:,0])/np.sum(np_mtx_output[:,1])))
print("Perr2 = {0}".format(np.sum(np_mtx_output[:,2])/np.sum(np_mtx_output[:,3])))

#close fasta and bam files
BY_file.close()
RM_file.close()
