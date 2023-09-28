#!~/src/anaconda2/bin/python2.7

import numpy as np
import pandas as pd
import sys
import os
import time
from multiprocessing import Pool, TimeoutError
from contextlib import closing

#Main workspace
the_main_workspace = sys.argv[1]
#Number of chromosomes analyzed simulataneously
the_nb_splits = int(sys.argv[2])
nb_chromosomes = 16

def run_get_genotypes_per_chr_id(current_id_chr):
    os.system(("/hive1/arnaud.nguessan/src/anaconda2/bin/python2.7 /hive1/arnaud.nguessan/src/split_bam.py {0} /hive1/arnaud.nguessan/research/projects/yeast mapped_and_filtered_RM_reads.bam lst_barcodes_with_expression_data.txt Cell_CB_Z chr%02d"%(current_id_chr+1,)).format(the_main_workspace))
    print("All cells genotypes obtained for chromosome {0}!".format(current_id_chr))

# start pool of parallel worker processes
with closing(Pool(processes=the_nb_splits)) as pool:
    pool.map(run_get_genotypes_per_chr_id, np.arange(nb_chromosomes).tolist())
    pool.terminate()
