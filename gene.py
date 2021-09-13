from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import predict
import os

# ararnge the data such that gene locations are indexed for each cheomosome, can also add a column of index as fraction of the whole 
def arrange_by_index(df,col='index',newcol=None):
    i = 2
    max_index = []  # how many observations per chromosome
    j=0
    for j in range(1, len(df)):
        if df['chromosome'][j] != df['chromosome'][j - 1]:
            max_index.append(i-1)
            i = 1
        df.loc[j, ['index']] = [i]
        i += 1
    #    j += 1
    max_index.append(i-1)
    df.loc[0, ['index']] = [1] # index now represents the relative location of the gene of it's chromosome

    if newcol is None: 
        return df
    i = 0
    df.loc[0, [newcol]] = [df[col][0] / max_index[0]]
    for j in range(1, len(df)):
        if df['chromosome'][j] != df['chromosome'][j - 1]:
            i += 1
        df.loc[j, [newcol]] = [df[col][j] / max_index[i]]
    return df


# input: data frame of genes locations on chromosomes
# output: the dataframe with added column of the gap on the chromosome where gene isn't mapped
#         works on the dataframe itself, not on a copy
def calc_unknown_gap(df, colname = 'gap'):
    for j in range(1, len(df)):
        if df['chromosome'][j] != df['chromosome'][j - 1]:
            df.loc[j, [colname]] = df['start'][j]
        else:
            tmp = df['start'][j] - df['end'][j-1]
            if tmp < 0: tmp = 0
            df.loc[j, [colname]] = tmp
    df.loc[0, [colname]] = df['start'][0]
    return df


# ararnge the data such that gene locations are fractions of the chromosome
def gene_as_fraction(data,col,newcol):
    df = data.copy()
    length = []
    for j in range(1, len(df)):
        if j == len(df) - 1 or df['chromosome'][j] != df['chromosome'][j + 1]:
            length.append(df['end'][j])

    i = 0
    df.loc[0, [newcol]] = [df[col][0] / length[0]]
    for j in range(1, len(df)):
        if df['chromosome'][j] != df['chromosome'][j - 1]:
            i += 1
        df.loc[j, [newcol]] = [df[col][j] / length[i]]
    return df

def split_train_test_draft(df, percent):
    train, test = train_test_split(df, train_size=percent, random_state=None, shuffle=True, stratify=None)
    train_df = {}
    test_df = {}
    chromosomes = df['chromosome'].unique()
    for crm in chromosomes:
        train_df[crm] = train.loc[train['chromosome'] == crm].drop(['chromosome'], axis=1)
        test_df[crm] = test.loc[test['chromosome'] == crm].drop(['chromosome'], axis=1)
        test.loc[test['chromosome'] == crm].drop(['chromosome'], axis=1)
    return train, test, train_df, test_df


# add columns of different clusters by method, the file name is expected to
#  contain the number of clusters first followes by '_'
def add_clusters(files_list, df, method):
    ## Merge original table with other tables to add other clustering columns
    data = df.copy()
    for file in files_list:
        ndf = pd.read_csv(file)[['gene', 'cluster']]
        name = os.path.basename(file)
        name = name.split(".")[0]
        header = name.split("_")
        data = pd.merge(
            data,
            ndf,
            how="left",
            on=['gene'],
            #left_index=True,
            #right_index=False,
            sort=False,
            suffixes=("", f'_{header[0]}_{method}'),
            copy=True,
            indicator=False,
            validate=None,
        )
    return data

# counts how any genes sticked together in all clustering methods
def count_loyal_genes(data ,method = ''):
    columns = []
    df = data.copy()
    if method == '':
        for col in df.columns:
            if 'cluster_' in col:
                columns.extend([col])
    else:
        for col in df.columns:
            if f'_{method}' in col:
                columns.extend([col])
    df = data[['gene']+columns]
    #
    df['factorize'] = 0
    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
             67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
             139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
             223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
             293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373,379,
             383, 389, 397, 401]
    i = 0
    for col in columns:
        df['factorize'] = df['factorize'] + df[col] *prime[i]
        i = i + 1
    return(Counter(df['factorize']))
   
# plot cluster view
# input: dataframe, chromosome index (not name), header of the cluster column 
def cluster_pattern(df,crm,cluster='cluster'):
    chromosomes = df['chromosome'].unique()
    seq = list(df[df['chromosome'] == chromosomes[crm]][cluster])
    flag = False
    lines = {}
    j = 0
    for i in range(1,len(seq)):
        if (seq[i] == seq[i-1]):
            if flag == True:
                lines[j].extend([seq[i]])
            if flag == False:
                flag = True
                j = i -1
                lines[j] = [seq[i-1],seq[i]]
        else:
            flag = False
    return lines






