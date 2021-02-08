#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_assembly(table):
    table = pd.read_csv('/Users/valery/Desktop/nextseq_auto_500/sequencing_assembly_report.txt', sep = '\t')
    table['ISOLATE'] = table.ISOLATE.str.split('_')
    table['ISOLATE'] = np.where(table['ISOLATE'].str.len().isin({2,3}),table['ISOLATE'].str[1] , table['ISOLATE'].str[0])
    table['ORGID'] = table['ISOLATE'].str.split('-').str[0]
    table['type'] = table['ISOLATE'].str.split('-', 1).str[1]
    table['DEPTH'] = table['DEPTH'].str.replace('X','')
    cols = table.columns.drop(['ORGID', 'type', 'ISOLATE'])
    table[cols] = table[cols].apply(pd.to_numeric, errors='coerce')
    return table
data_new = parse_assembly(data_ass)

# The Pseudomonas aeruginosa genome (G + C content 65–67%, size 5.5–7 Mbp) is made up of a single circular chromosome and a variable number of plasmids.
# https://www.frontiersin.org/articles/10.3389/fmicb.2011.00150/full
# Acinetobacter baumannii:
# median total length (Mb): 3.97473
#  median protein count: 3692
#  median GC%: 39
#  Klebsiella pneumoniae
#  median total length (Mb): 5.59628
#  median protein count: 5330
#  median GC%: 57.2
#  Escherichia-coli
#  4.5 to 5.5 Mb
#  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC96301/
#  median total length (Mb): 5.12122
#  median protein count: 4739
#  median GC%: 50.6
#  Serratia marcescens
#  median total length (Mb): 5.21305
#  median protein count: 4829
#  median GC%: 59.7775
#  Enterobacter-cloacae
#  median total length (Mb): 5.05826
#  median protein count: 4690
#  median GC%: 55
#  Staphylococcus-aureus
#  median total length (Mb): 2.83686
#  median protein count: 2690
#  median GC%: 32.7
#  
#  
data_new['len_alarm'] = 0
data_new['gc_alarm'] = 0
data_new.loc[(data_new['type'] == 'Pseudomonas-aeruginosa') & ((data_new['GenomeLength'] < 5500000) |(data_new['GenomeLength'] > 7500000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Pseudomonas-aeruginosa') & ((data_new['GC'] < 65) |(data_new['GC'] > 67)), 'gc_alarm'] = 1
data_new.loc[(data_new['type'] == 'Acinetobacter-baumannii') & ((data_new['GenomeLength'] < 3500000) |(data_new['GenomeLength'] > 5000000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Acinetobacter-baumannii') & ((data_new['GC'] < 38) |(data_new['GC'] > 40)), 'gc_alarm'] = 1
data_new.loc[(data_new['type'] == 'Klebsiella-pneumonia') & ((data_new['GenomeLength'] < 5100000) |(data_new['GenomeLength'] > 6600000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Klebsiella-pneumonia') & ((data_new['GC'] < 56.2) |(data_new['GC'] > 58.2)), 'gc_alarm'] = 1
data_new.loc[(data_new['type'] == 'Escherichia-coli') & ((data_new['GenomeLength'] < 4600000) |(data_new['GenomeLength'] > 6100000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Escherichia-coli') & ((data_new['GC'] < 49.6) |(data_new['GC'] > 51.6)), 'gc_alarm'] = 1
data_new.loc[(data_new['type'] == 'Serratia-marcescens') & ((data_new['GenomeLength'] < 4700000) |(data_new['GenomeLength'] > 6200000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Serratia-marcescens') & ((data_new['GC'] < 58.7) |(data_new['GC'] > 60.7)), 'gc_alarm'] = 1
data_new.loc[(data_new['type'] == 'Enterobacter-cloacae') & ((data_new['GenomeLength'] < 4500000) |(data_new['GenomeLength'] > 6000000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Enterobacter-cloacae') & ((data_new['GC'] < 54) |(data_new['GC'] > 56)), 'gc_alarm'] = 1
data_new.loc[(data_new['type'] == 'Staphylococcus-aureus') & ((data_new['GenomeLength'] < 2300000) |(data_new['GenomeLength'] > 3800000)), 'len_alarm'] = 1
data_new.loc[(data_new['type'] == 'Staphylococcus-aureus') & ((data_new['GC'] < 31.7) |(data_new['GC'] > 33.7)), 'gc_alarm'] = 1
#check gc and len
data_new[data_new['gc_alarm'] == 1]
type_bac = data_ass['tax'].value_counts()
type_bac = type_bac.reset_index()
type_bac.columns = ['type', 'count']
#type_bac = type_bac.set_index('type')

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = type_bac['type']
sizes = type_bac['count']
explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1 = plt.figure(figsize=(7,6))
ax1 = fig1.add_subplot()
#ax1.set_position([0.1,0.3,0,0.3])
#fig1.tight_layout()
#subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)
#fig1.subplots_adjust(0.1,0.3,0.9,0.7) -- 2

fig1.subplots_adjust(0.125,0.2,0.9,0.8)
#fig1.subplots_adjust(0.125, 0.1,0.9, 0.9, 0.2,0.2)
#fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=10, radius = 3)
#ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)    
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#ax1.set_title("Рисунок 1.Распределение образцов по видам")
plt.savefig('types_bac_pie_7.png')
plt.show()
