

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

def process_seqlist(all_seqs,prefilename):
    ''' helping chunk '''
    list_of_seqs_lens=[]
    list_of_seqs=[]
    print('starting while loop')
    if os.path.exists(prefilename[:-4]+'_temp_ALLSEQS.npy'):
        print('loading existing temp file')
        all_seqs=np.load(prefilename[:-4]+'_temp_ALLSEQS.npy')
        predf=pd.read_csv(prefilename[:-4]+'_temp.csv')
        list_of_seqs=list(predf.list_of_seqs)
        list_of_seqs_lens=list(predf.n_similar)
        print('list ',len(list_of_seqs),'lens ',len(list_of_seqs_lens),'all seqs', len(all_seqs))
    filelen=len(all_seqs)
    if len(all_seqs)!=0:
        while len(all_seqs)>1:
            if len(all_seqs)%100==0:
                print('on {} of {}'.format(len(all_seqs),filelen))
                predf=pd.DataFrame()
                predf['list_of_seqs']=list_of_seqs
                predf['n_similar']=list_of_seqs_lens
                predf['groups']=0
                np.save(prefilename[:-4]+'_temp_ALLSEQS.npy',np.array(all_seqs))
                predf.to_csv(prefilename[:-4]+'_temp.csv')
            seq = all_seqs[0]# add a sequence
            bool_list=[]
            for i in np.arange(0,len(all_seqs)):
            # if 97% or better match, add a number for the n_val
                        if fuzz.ratio(seq,all_seqs[i]) >95:
                             bool_list.append(False)
                        else:
                            bool_list.append(True)
            num_of_seqs = len(bool_list) - np.sum(bool_list)
            if num_of_seqs > 15:
                       list_of_seqs.append(seq)
                       list_of_seqs_lens.append(num_of_seqs)
            all_seqs= list(compress(all_seqs,bool_list))
    predf=pd.DataFrame()
    predf['list_of_seqs']=list_of_seqs
    predf['n_similar']=list_of_seqs_lens
    predf['groups']=0
    predf.to_csv(prefilename)
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

