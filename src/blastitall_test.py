

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
import os, csv, glob, sys
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

def process_seqlist(all_seqs, prefilename,
                    max_mismatch=5,
                    anchors=(0, 10, 25, 50, 75),
                    checkpoint_every=5000):
    
        # --- sanitize input sequences ---
    clean_seqs = []
    bad = 0

    for s in all_seqs:
        if not isinstance(s, str):
            bad += 1
            continue
        s = s.strip().upper()
        # remove anything not ACGTN
        s = ''.join(c if c in 'ACGTN' else 'N' for c in s)
        if len(s[:100]) != 100:
            bad += 1
            continue

        clean_seqs.append(s)

    print(f"using {len(clean_seqs)} clean sequences ({bad} dropped)")
    all_seqs = clean_seqs

    all_seqs = clean_seqs


    # --- DNA encoding table (FIXED) ---
    table = np.full(256, 4, dtype=np.uint8)
    table[ord('A')] = 0
    table[ord('C')] = 1
    table[ord('G')] = 2
    table[ord('T')] = 3
    table[ord('N')] = 4

    def encode(seq):
        if not isinstance(seq, str):
            raise ValueError(f"Non-string sequence encountered: {type(seq)}")
        return table[np.frombuffer(seq.encode(), dtype=np.uint8)]


    list_of_seqs = []
    list_of_seqs_lens = []

    # --- restore checkpoint if present ---
    temp_npy = prefilename[:-4] + '_temp_ALLSEQS.npy'
    temp_csv = prefilename[:-4] + '_temp.csv'

    if os.path.exists(temp_npy) and os.path.exists(temp_csv):
        print('loading existing temp file')
        encoded = np.load(temp_npy)
        predf = pd.read_csv(temp_csv)
        list_of_seqs = list(predf.list_of_seqs)
        list_of_seqs_lens = list(predf.n_similar)
    else:
        encoded = np.array([encode(s) for s in all_seqs], dtype=np.uint8)

    total = len(encoded)
    used = np.zeros(total, dtype=bool)

    print(f"starting hamming clustering on {total} sequences")

    # --- anchor buckets ---
    buckets = defaultdict(list)
    for i, seq in enumerate(encoded):
        key = tuple(seq[a] for a in anchors)
        buckets[key].append(i)

    processed = 0

    # --- clustering ---
    for indices in buckets.values():
        indices = np.array(indices, dtype=int)

        for idx in indices:
            if used[idx]:
                continue

            ref = encoded[idx]

            diffs = (encoded[indices] != ref) & (encoded[indices] != 4) & (ref != 4)
            mismatches = np.sum(diffs, axis=1)

            close_mask = mismatches <= max_mismatch
            close_indices = indices[close_mask]

            num_of_seqs = len(close_indices)

            if num_of_seqs > 15:
                list_of_seqs.append(''.join('ACGTN'[b] for b in ref))
                list_of_seqs_lens.append(num_of_seqs)

            used[close_indices] = True
            processed += len(close_indices)

            # --- checkpoint ---
            if processed % checkpoint_every == 0:
                print(f"processed {processed} / {total}")
                np.save(temp_npy, encoded[~used])

                predf = pd.DataFrame({
                    'list_of_seqs': list_of_seqs,
                    'n_similar': list_of_seqs_lens,
                    'groups': 0
                })
                predf.to_csv(temp_csv, index=False)

    # --- final output ---
    predf = pd.DataFrame({
        'list_of_seqs': list_of_seqs,
        'n_similar': list_of_seqs_lens,
        'groups': 0
    })
    predf.to_csv(prefilename, index=False)

    return predf

def get_ncbi(file_path, output_fld):
    """
    Process sequences from a TSV file, cluster them using Hamming distance,
    and save representative sequences with grouping info.
    """

    try:
        file = pd.read_table(file_path)
        # calculate median quality
        file['medqual'] = [np.median(eval(val)) for val in file.QUALITY]
        # select sequences with median quality > 32
        qual_seqs = file.NUC_SEQ[file.medqual > 32].tolist()
    except Exception as e:
        print(f"{file_path} FAILED: {e}")
        return

    prefilename = f'{output_fld}/{os.path.basename(file_path).split(".tsv")[0]}_pre.csv'

    # --- determine processing step ---
    step = 0
    predf = None

    if os.path.exists(prefilename):
        if os.path.exists(prefilename[:-4] + '_2.csv'):
            print(f'{prefilename} has already been through all pre-processing! starting blasts')
            predf = pd.read_csv(prefilename[:-4] + '_2.csv', index_col=0)
            step = 2
        else:
            print(f'{prefilename} already has been processed! using pre-made file')
            predf = pd.read_csv(prefilename, index_col=0)
            step = 1

    # --- Step 0: cluster sequences ---
    if step < 1:
        print(f"Step 0: clustering sequences for {file_path}")
        predf = process_seqlist(qual_seqs, prefilename)  # returns predf
        step = 1

    # --- Step 1 -> 2: assign groups ---
    if step < 2:
        print(f"Step 1 -> 2: assigning groups for {file_path}")
        group_num = 0
        ns_in_seq = []

        # ensure predf has correct columns
        predf.columns = [c.strip() for c in predf.columns]
        if 'list_of_seqs' not in predf.columns or 'n_similar' not in predf.columns:
            raise RuntimeError(f"{prefilename} has unexpected columns: {predf.columns}")

        # compute Ns per sequence
        for seq in predf['list_of_seqs']:
            ns_in_seq.append(seq.count('N'))

        predf['ns_in_seq'] = ns_in_seq
        predf['groups'] = 0

        for idx in predf.index:
            if predf.at[idx, 'groups'] == 0:
                group_num += 1
                predf.at[idx, 'groups'] = group_num
                seq = predf.at[idx, 'list_of_seqs']
                seq_n = seq.count('N')

                for i in predf.index[idx:]:
                    comp_seq = predf.at[i, 'list_of_seqs']
                    comp_n = comp_seq.count('N')
                    # Hamming similarity + Ns check
                    mismatches = sum(c1 != c2 and c1 != 'N' and c2 != 'N'
                                     for c1, c2 in zip(seq, comp_seq))
                    if mismatches + max(seq_n, comp_n) <= 5:  # ~95% threshold
                        predf.at[i, 'groups'] = group_num

        predf.to_csv(prefilename[:-4] + '_2.csv')

    # --- Step 2: generate representative sequences ---
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
    print(f"Finished processing {file_path}, info saved to {infofilename}")

    
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
    for file in file_paths:
        print(' on ', file)
        get_ncbi(file,output_fld)
    print('done')

