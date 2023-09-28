"""
MIT License
Copyright (c) 2021 Arnaud N'Guessan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#Script for extracting SPECIFIED BARCODES genotype in a SPECIFIC CHROMOSOME using a multi-barcode bam file
#import libraries
from pathlib import Path
import pysam
import sys
import os 
import numpy as np
import pandas as pd

#import reference data and script argument
workspace_path = sys.argv[1]
yeast_project_wp_path = sys.argv[2]
the_input_bam=sys.argv[3]
the_barcodes_file=sys.argv[4]
the_output_prefix=sys.argv[5]
the_contig=sys.argv[6]
df_pos_snps = pd.read_csv("{0}/BYxRM_nanopore_SNPs.gd".format(yeast_project_wp_path),sep="\t",header=None,dtype={ '0': str, '1': str, '2': int, '3': str })
df_pos_snps.columns = ["mutation", "chromosome","position","Allele"]
df_pos_snps = df_pos_snps[df_pos_snps.chromosome==the_contig]
df_pos_snps["the_key"] = ["{0}_{1}".format(df_pos_snps.chromosome.tolist()[c],df_pos_snps.position.tolist()[c]) for c in np.arange(np.shape(df_pos_snps)[0])]

BY_file = pysam.FastaFile("{0}/BY4742_fixed.fa".format(yeast_project_wp_path))

#Create dataframes of RM and BY alleles count
df_lst_cells = pd.read_csv(the_barcodes_file,sep="\t",header=None,dtype={ '0': str })
df_lst_cells.columns = ["cell"]
lst_cells = df_lst_cells["cell"].tolist()
nb_cells = np.shape(df_lst_cells)[0]
df_BY_count = pd.DataFrame(np.zeros((nb_cells,np.shape(df_pos_snps)[0])),dtype=int,index=lst_cells,columns=[str(x) for x in df_pos_snps["position"]])
df_RM_count = pd.DataFrame(np.zeros((nb_cells,np.shape(df_pos_snps)[0])),dtype=int,index=lst_cells,columns=[str(x) for x in df_pos_snps["position"]])
lst_label_chromosomes = ["chr%02d"%(i,) for i in (np.arange(16)+1)]

#Function that update cells genotype based on mapped reads
def modify_cells_genotype_from_pileups_at_position(pileups, contig, the_pos):
    global df_BY_count
    global df_RM_count
    RM_allele = (df_pos_snps[(df_pos_snps["chromosome"]==contig) & (df_pos_snps["position"]==the_pos)])["Allele"].tolist()[0]
    BY_allele = BY_file.fetch(contig)[the_pos]
    for pileupread in pileups:
        if not pileupread.is_del and not pileupread.is_refskip:
            try:
                barcode = pileupread.alignment.get_tag("CB")
                current_allele = pileupread.alignment.query_sequence[pileupread.query_position]
                if current_allele == RM_allele:
                    df_RM_count.at[barcode,str(the_pos)] = df_RM_count.at[barcode,str(the_pos)] + 1 
                elif current_allele == BY_allele:
                    df_BY_count.at[barcode,str(the_pos)] = df_BY_count.at[barcode,str(the_pos)] + 1
                else:
                    continue 
            except KeyError:
                pass
def main(input_bam, barcodes_file, output_prefix, contig):
    #import multicell bam file
    alignment = pysam.AlignmentFile("{0}/data/{1}".format(workspace_path,input_bam))
    #taking into account that you can pass a file with the list of barcodes or a single barcode as the input argument ""
    if Path(barcodes_file).is_file():
        with open(barcodes_file, "r") as fh:
            barcodes = [l.rstrip() for l in fh.readlines()]
    else:
        barcodes = [barcodes_file]
        print("Extracting single barcode: {0}".format(barcodes))

    for current_pos_in_chr in df_pos_snps.position.tolist():
        print("Analysis started for position {0} in chromosome {1}".format(current_pos_in_chr,contig))
        for pileupcolumn in alignment.pileup(contig=contig,start=current_pos_in_chr-1,stop=current_pos_in_chr,truncate=1):
            #print("coverage = {0}".format(pileupcolumn.n))
            #print("position is {0}".format(pileupcolumn.pos+1))
            pileups = pileupcolumn.pileups
            try:
                modify_cells_genotype_from_pileups_at_position(pileups=pileups,contig=contig, the_pos=current_pos_in_chr)
            except KeyError:
                pass

if __name__ == "__main__":
    import sys

    main(
        input_bam=the_input_bam,
        barcodes_file=the_barcodes_file,
        output_prefix=the_output_prefix,
        contig=the_contig
    )
    #Create dataframe of RM_ratio
    df_RM_ratio = df_RM_count/(df_BY_count + df_RM_count)
    #save BY and RM counts + RM_ratio dataframes
    df_BY_count.to_csv("{0}/data/good_cells_genotypes/BY_count_{1}.csv".format(workspace_path,the_contig), sep='\t',na_rep="NA",header=True,index=True)
    df_RM_count.to_csv("{0}/data/good_cells_genotypes/RM_count_{1}.csv".format(workspace_path,the_contig), sep='\t',na_rep="NA",header=True,index=True)
    df_RM_ratio.to_csv("{0}/data/good_cells_genotypes/RM_ratio_{1}.csv".format(workspace_path,the_contig), sep='\t',na_rep="NA",header=True,index=True)

BY_file.close()
