

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ejdennis
purpose is to take in a string that is a path to a folder of csv files
from obitools and a place to save files, and then blast each unique
sequence and save the results

input:
1. a string, path to a folder full of video
2. a string, path an output directory (the parent folde rmust exist)

outputs:
1. csv files for each blast result, all will be saved in the output directory
"""

import numpy as np
import pandas as pd
import os, csv, glob, sys, time
import matplotlib.pyplot as plt
from Bio.Blast import NCBIWWW
from Bio import SeqIO
from Bio.Blast import NCBIXML
from thefuzz import fuzz
from itertools import compress
import multiprocessing

from collections import defaultdict
import numpy as np
import pandas as pd
import os
from collections import defaultdict
def process_seqlist(all_seqs):
    """
    Process sequences and select representative sequences based on Hamming similarity.
    Returns a DataFrame with list_of_seqs, n_similar, and groups.
    """

    # --- sanitize sequences ---
    clean_seqs = []
    for s in all_seqs:
        if not isinstance(s, str):
            continue
        s = s.strip().upper()
        s = ''.join(c if c in 'ACGTN' else 'N' for c in s)
        s=s[:100]
        if len(s) == 100:
            clean_seqs.append(s)

    all_seqs = clean_seqs
    print(f"Using {len(all_seqs)} sequences after cleaning")

    list_of_seqs = []
    list_of_seqs_lens = []

    while len(all_seqs) > 0:
        seq = all_seqs[0]
        bool_list = []

        # Hamming similarity check
        for other_seq in all_seqs:
            mismatches = sum(c1 != c2 and c1 != 'N' and c2 != 'N' for c1, c2 in zip(seq, other_seq))
            if mismatches <= 5:  # ~95% similarity
                bool_list.append(False)
            else:
                bool_list.append(True)

        num_similar = len(bool_list) - sum(bool_list)
        if num_similar > 15:
            list_of_seqs.append(seq)
            list_of_seqs_lens.append(num_similar)

        # Keep only sequences not grouped
        all_seqs = [s for s, keep in zip(all_seqs, bool_list) if keep]

    # Build DataFrame
    predf = pd.DataFrame({
        'list_of_seqs': list_of_seqs,
        'n_similar': list_of_seqs_lens,
        'groups': 0
    })

    return predf

def get_ncbi(file_path, output_fld):
    """
    Process a TSV file in steps with checkpointing:
    step 0: process sequences
    step 1->2: assign groups
    step 2: select representative sequences
    """

    try:
        file = pd.read_table(file_path)
        file['medqual'] = [np.median(eval(val)) for val in file.QUALITY]
        qual_seqs = file.NUC_SEQ[file.medqual > 32].tolist()
    except Exception as e:
        print(f"{file_path} FAILED: {e}")
        return

    prefilename = f'{output_fld}/{os.path.basename(file_path).split(".tsv")[0]}_pre.csv'
    step = 0
    predf = None

    # --- determine which step to start from ---
    if os.path.exists(prefilename):
        if os.path.exists(prefilename[:-4] + '_2.csv'):
            predf = pd.read_csv(prefilename[:-4] + '_2.csv', index_col=0)
            step = 2
            print(f"{prefilename} already fully processed")
        else:
            predf = pd.read_csv(prefilename, index_col=0)
            step = 1
            print(f"{prefilename} already processed step 0")

    # --- step 0: process sequences ---
    if step < 1:
        print(f"Step 0: processing sequences for {file_path}")
        predf = process_seqlist(qual_seqs)
        predf.to_csv(prefilename)
        step = 1

    # --- step 1 -> 2: assign groups ---
    if step < 2:
        print(f"Step 1->2: assigning groups for {file_path}")

        ns_in_seq = [seq.count('N') for seq in predf['list_of_seqs']]
        predf['ns_in_seq'] = ns_in_seq
        predf['groups'] = 0
        group_num = 0

        for idx in predf.index:
            if predf.at[idx, 'groups'] == 0:
                group_num += 1
                seq = predf.at[idx, 'list_of_seqs']
                seq_n = seq.count('N')
                predf.at[idx, 'groups'] = group_num

                for i in predf.index[idx:]:
                    comp_seq = predf.at[i, 'list_of_seqs']
                    comp_n = comp_seq.count('N')
                    mismatches = sum(c1 != c2 and c1 != 'N' and c2 != 'N'
                                     for c1, c2 in zip(seq, comp_seq))
                    if mismatches + max(seq_n, comp_n) <= 5:
                        predf.at[i, 'groups'] = group_num

        predf.to_csv(prefilename[:-4] + '_2.csv')
        step = 2

    # --- step 2: select representative sequences ---
    predf = pd.read_csv(prefilename[:-4] + '_2.csv', index_col=0)
    seqs = []
    n_similar = []
    indices_grouped = []

    for group in np.unique(predf.groups):
        subdf = predf[predf.groups == group].copy()
        min_ns = subdf.ns_in_seq.min()
        keep_idx = subdf[subdf.ns_in_seq == min_ns].index[0]

        seqs.append(subdf.at[keep_idx, 'list_of_seqs'])
        n_similar.append(subdf['n_similar'].sum())
        indices_grouped.append(list(subdf.index))

    df = pd.DataFrame({
        'seqs': seqs,
        'n_similar': n_similar,
        'indices_grouped': indices_grouped
    })

    infofilename = f'{output_fld}/{os.path.basename(file_path).split(".tsv")[0]}_info.csv'
    df.to_csv(infofilename, index=False)
    print(f"Finished {file_path}, info saved to {infofilename}")
    print('starting blast for file {}'.format(file_path))
    # save out as fasta!
    # read in fasta!
    result_list = []
    seq_counter=-1
    
    print('starting blast for file {}'.format(file_path))

    for seq_counter, seq in enumerate(seqs):

        filename_new = '{}/results_{}_{}.csv'.format(
            output_fld,
            file_path.split('/')[-1].split('.tsv')[0],
            seq_counter
        )

        if os.path.isfile(filename_new):
            print(f'already processed seq {seq_counter} of {len(seqs)}, skipping')
            continue

        print(f'starting blast {seq_counter} of {len(seqs)}')

        blast_xml = safe_qblast(seq)

        # ---- BLAST failed completely ----
        if blast_xml is None:
            print(f"[SKIPPING] seq {seq_counter} due to BLAST failure")
            continue

        results_filename = os.path.join(
            output_fld,
            "results_{}_{}.xml".format(
                file_path.split('/')[-1].split('.tsv')[0],
                seq_counter
            )
        )

        with open(results_filename, 'w') as save_file:
            save_file.write(blast_xml)

        data_tuples = []

        try:
            for record in NCBIXML.parse(open(results_filename)):
                if record.alignments:
                    for align in record.alignments:
                        for hsp in align.hsps:
                            if hsp.expect < 1e-20:
                                data_tuples.append(
                                    (align.hit_def,
                                    align.accession,
                                    hsp.sbjct,
                                    hsp.identities,
                                    hsp.expect)
                                )
        except Exception as e:
            print(f"[XML PARSE ERROR] seq {seq_counter}: {e}")
            continue

        pd.DataFrame(
            data_tuples,
            columns=['hit_definition','hit_accession','subject','identities','expect']
        ).to_csv(filename_new, index=False)

'''    
for seq in seqs:
        seq_counter+=1
        filename_new='{}/results_{}_{}.csv'.format(output_fld, file_path.split('/')[-1].split('.tsv')[0],seq_counter)
        if os.path.isfile(filename_new):
            print('already processed seq {} of {}, skipping'.format(seq_counter,len(seqs)))
        else:
            print('starting blast {} of {}'.format(seq_counter,len(seqs)))
            result_handle = NCBIWWW.qblast('blastn','nt',seq)
            results_filename = os.path.join(output_fld,"results_{}_{}.xml".format(file_path.split('/')[-1].split('.tsv')[0],seq_counter))
            data_tuples=[]
            with open(results_filename, 'w') as save_file:
                blast_results = result_handle.read()
                save_file.write(blast_results)
            for record in NCBIXML.parse(open(results_filename)):
                if record.alignments:
                    for align in record.alignments:
                        for hsp in align.hsps:
                            if hsp.expect < 1e-20:
                                data_tuples.append((align.hit_def,align.accession,hsp.sbjct,hsp.identities,hsp.expect))
            pd.DataFrame(data_tuples,columns=['hit_definition','hit_accession','subject','identities','expect']).to_csv(filename_new)
    print('saving info file for {}'.format(infofilename))
'''
    
if __name__ == "__main__":

    print(sys.argv)
    # deal with inputs
    try:
        obi_out_fld=str(sys.argv[1])
    except:
        print("this function requires two inputs: the first must be ",
                    "a string that leads to a folder of csvs. you did not enter",
                    "a string, your sys.argvs are: {}".format(sys.argv))
    try:
                output_fld=str(sys.argv[2])
                os.path.isdir(os.path.dirname(output_fld))
    except:
                print("this function requires two inputs: the second must be ",
                    "a string that leads to a folder of csvs. you did not enter",
                    "a string, your sys.argvs are: {}".format(sys.argv))


    file_paths = [os.path.join(obi_out_fld,file) for file in os.listdir(obi_out_fld)]
    
    pool= multiprocessing.Pool(20)
    pool.starmap(get_ncbi,[(file,output_fld) for file in file_paths])
    print('done')

