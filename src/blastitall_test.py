

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
    '''updated 20250605'''
    '''updated 20260108 to deal with Ns'''
    try:
        file = pd.read_table(file_path)
        file['medqual']=[np.median(eval(val)) for val in file.QUALITY]
        qual_seqs= np.array(file.NUC_SEQ[file.medqual>32])
    except:
        print('{} FAILED'.format(file_path))
    step=0
    prefilename='{}/{}_pre.csv'.format(output_fld,file_path.split('/')[-1].split('.tsv')[0])
    # look for processed files, set step accordingly
    if os.path.exists(prefilename):
        if os.path.exists(prefilename[:-4]+'_2.csv'):
            print('{} has already been through all pre-processing! starting blasts'.format(prefilename))
            predf=pd.read_csv(prefilename[:-4]+'_2.csv',index_col=0)
            step=2
        else:
            print('{} already has been processed! using pre-made file'.format(prefilename))
            predf=pd.read_csv(prefilename,index_col=0)
            step=1
    # start processing
    if step < 1:
	    list_of_seqs_lens=[]
	    list_of_seqs=[]
	    print('starting while loop for file_path {}'.format(file_path))
	    process_seqlist(qual_seqs,prefilename)
	    predf=pd.read_csv(prefilename,index_col=0)
    if step < 2:
            print('starting from pre file for {}'.format(file_path))
            group_num=0
            ns_in_seq=[]
            for idx in predf.index:
                seq = str(predf.list_of_seqs[idx])
                seq_count=seq.count('n')
                ns_in_seq.append(seq_count)
                if predf.groups[idx]==0:
                    group_num+=1
                    predf.loc[idx,'groups']=group_num
                    for i in predf.index[idx:]:
                        if fuzz.ratio(seq,predf.list_of_seqs[i])+np.max([seq_count,str(predf.list_of_seqs[i]).count('n')])>97:
                            predf.loc[i,'groups']=group_num
            predf['ns_in_seq']=ns_in_seq
            predf.to_csv(prefilename[:-4]+'_2.csv')
    predf=pd.read_csv(prefilename[:-4]+'_2.csv',index_col=0)
    print('starting from pre_2 file')
    seqs=[]
    n_similar=[]
    indices_grouped=[]
    for group in np.unique(predf.groups):
        subdf=predf[predf.groups==group].copy()
        if len(subdf[subdf.ns_in_seq==np.min(subdf.ns_in_seq)])==1:
            keep_index=subdf.index[subdf.ns_in_seq==np.min(subdf.ns_in_seq)]
        else:
            keep_index = subdf.index[subdf.ns_in_seq==np.min(subdf.ns_in_seq)][0]
        seqs.append(str(subdf.list_of_seqs[subdf.ns_in_seq==np.min(subdf.ns_in_seq)].values[0]))
        n_similar.append(np.sum([val for val in subdf.n_similar]))
        indices_grouped.append(list(subdf.index))
    df=pd.DataFrame()
    df['seqs']=seqs
    df['n_similar']=n_similar
    df['indices_grouped']=indices_grouped
    infofilename='{}/{}_info.csv'.format(output_fld,file_path.split('/')[-1].split('.tsv')[0])
    df.to_csv(infofilename)
    
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

