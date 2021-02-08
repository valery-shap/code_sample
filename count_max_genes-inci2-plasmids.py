#!/usr/bin/env python
# coding: utf-8
from Bio import SeqIO
import pandas as pd
def true_id(text):
    first = text.split('|')
    if len(first) > 1:
        true = first[1]
    else:
        true = text
    return true 
blastn = pd.read_csv('/Users/valery/Desktop/inci2_db/find_genes/find_all.tsv', sep = '\t', names = [1,2,3,4,5,6,7,8,9,10,11])
blastn = blastn.reset_index()
blastn[12] = blastn[1].apply(true_id)
blastn = blastn[blastn[12].isin(ids_full) == True]
blastn_2 =blastn.groupby(12)[['index', 5, 6]].apply(lambda x: x.set_index('index').T.to_dict('list')).to_dict()
id_seq = {}
for seq_record in SeqIO.parse("/Users/valery/Desktop/inci2_db/plasmids_db_csp.fasta", "fasta"):
    id_seq[seq_record.id] = seq_record.seq
for id_, genes in blastn_2.items():
    for gene, coord in genes.items():
        if id_ in id_seq.keys():
            if coord[0] < coord[1]:
                seq = id_seq[id_][coord[0]-1:coord[1]]
                blastn_2[id_][gene].append(seq)
            else:
                seq = id_seq[id_][coord[1]-1:coord[0]]
                seq = seq.reverse_complement()
                blastn_2[id_][gene].append(seq)
for id_, genes in blastn_2.items():
    seq = ''
    for gene, coord in genes.items():
        seq += coord[2]
    blastn_2[id_]['seq'] = seq
sequences = ''
for id_, genes in blastn_2.items():
    sequences = sequences+'>' + str(id_)+ '\n' + str(blastn_2[id_]['seq']) + '\n'
open( "final.txt", "w+").write(sequences)
blastn_frame = pd.DataFrame.from_dict(blastn_2, orient = 'index')
blastn_frame = blastn_frame.reset_index()
blastn_seq = blastn_frame[['index', 'seq']]
blastn_seq_u = blastn_seq.drop_duplicates(subset = 'seq', keep = 'last')
sequences_2=''
for index, row in blastn_seq_u.iterrows():
    sequences_2 = sequences_2+'>' + str(row['index'])+ '\n' + str(row['seq']) + '\n'
sequences_2
open( "final_4.txt", "w+").write(sequences_2)
