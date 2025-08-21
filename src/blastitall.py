

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


def get_ncbi(file_path, output_fld):
    '''updated 20250605'''
    try:
        file = pd.read_table(file_path)
        file['medqual']=[np.median(eval(val)) for val in file.QUALITY]
        all_seqs= np.array(file.NUC_SEQ[file.medqual>32])
    except:
        print('{} FAILED'.format(file_path))
    list_of_seqs_lens=[]
    list_of_seqs=[]
    print('starting while loop for file_path {}'.format(file_path))
    if len(all_seqs)!=0:
        while len(all_seqs)>1:
            seq = all_seqs[0]# add a sequence
            bool_list=[]
            # get similar sequences, delete from the all_seqs list before iterating
            for i in np.arange(0,len(all_seqs)):
                    # if 97% or better match, add a number fo the n_val
                    if fuzz.ratio(seq,all_seqs[i]) >96:
                        bool_list.append(False)
                    else:
                        bool_list.append(True)
            num_of_seqs = len(bool_list) - np.sum(bool_list)
            if num_of_seqs > 15:
                list_of_seqs.append(seq)
                list_of_seqs_lens.append(num_of_seqs)
            all_seqs = list(compress(all_seqs,bool_list))
    # save out as fasta!
    # read in fasta!
    result_list = []
    seq_counter=-1
    for seq in list_of_seqs:
        seq_counter+=1
        filename_new='{}/results_{}_{}.csv'.format(output_fld, file_path.split('/')[-1].split('.csv')[0],seq_counter)
        if os.path.isfile(filename_new):
            print('already processed seq {} of {}, skipping'.format(seq_counter,len(list_of_seqs)))
        else:
            print('starting blast {} of {}'.format(seq_counter,len(list_of_seqs)))
            result_handle = NCBIWWW.qblast('blastn','nt',seq)
            results_filename = os.path.join(output_fld,"results_{}_{}.xml".format(file_path.split('/')[-1].split('.csv')[0],seq_counter))
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
    infofilename='{}/{}_info.csv'.format(output_fld,file_path.split('/')[-1].split('.csv')[0])
    print('saving info file for {}'.format(infofilename))
    infodf=pd.DataFrame({'list_of_seqs':list_of_seqs,'n_similar':list_of_seqs_lens}).to_csv(infofilename)

def get_old_ncbi(file_path, output_fld):
    '''depreciated 20250605'''
    try:
        file = pd.read_table(file_path)
        all_seqs=np.array(file.NUC_SEQ)
    except:
        print('{} FAILED'.format(file_path))
    list_of_seqs_lens=[]
    list_of_seqs=[]
    print('starting while loop for file_path {}'.format(file_path))
    if len(all_seqs)!=0:
        while len(all_seqs)>1:
            seq = all_seqs[0]# add a sequence
            bool_list=[]
            # get similar sequences, delete from the all_seqs list before iterating
            for i in np.arange(0,len(all_seqs)):
                    # if 97% or better match, add a number fo the n_val
                    if fuzz.ratio(seq,all_seqs[i]) >96:
                        bool_list.append(False)
                    else:
                        bool_list.append(True)
            num_of_seqs = len(bool_list) - np.sum(bool_list)
            if num_of_seqs > 15:
                list_of_seqs.append(seq)
                list_of_seqs_lens.append(num_of_seqs)
            all_seqs = list(compress(all_seqs,bool_list))
    # save out as fasta!
    # read in fasta!
    result_list = []
    seq_counter=-1
    for seq in list_of_seqs:
        seq_counter+=1
        filename_new='{}/results_{}_{}.csv'.format(output_fld, file_path.split('/')[-1].split('.csv')[0],seq_counter)
        if os.path.isfile(filename_new):
            print('already processed seq {} of {}, skipping'.format(seq_counter,len(list_of_seqs)))
        else:
            print('starting blast {} of {}'.format(seq_counter,len(list_of_seqs)))
            result_handle = NCBIWWW.qblast('blastn','nt',seq)
            results_filename = os.path.join(output_fld,"results_{}_{}.xml".format(file_path.split('/')[-1].split('.csv')[0],seq_counter))
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
    infofilename='{}/{}_info.csv'.format(output_fld,file_path.split('/')[-1].split('.csv')[0])
    print('saving info file for {}'.format(infofilename))
    infodf=pd.DataFrame({'list_of_seqs':list_of_seqs,'n_similar':list_of_seqs_lens}).to_csv(infofilename)
    
def get_input_list(list_of_paths, output_fld):
    input_list=[(file, output_fld) for file in list_of_paths]
    return input_list

if __name__ == "__main__":

    print(sys.argv)
    # deal with inputs\
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

    chunk_list=np.arange(0,int(np.floor(len(file_paths)/10)))
    chunk_remainder = len(file_paths)%10

    for chunk_val in chunk_list[:-1]:
        pool = multiprocessing.Pool(36)
        pool.starmap(get_ncbi,get_input_list([file for file in file_paths[int(chunk_val*10):int(chunk_val*10)+10]],output_fld))
    if chunk_remainder > 0:
        if chunk_remainder > 1:
            pool = multiprocessing.Pool(36)
            pool.starmap(get_ncbi,get_input_list([file for file in file_paths[int(-1*(chunk_remainder-1)):]],output_fld))
        else:
            pool = multiprocessing.Pool(36)
            get_ncbi(file_paths[-1], output_fld)
    print('done')
            
            
            
            
            
            
