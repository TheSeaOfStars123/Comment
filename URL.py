# @Time : 2021/7/4 11:16 PM 
# @Author : zyc
# @File : URL.py 
# @Title :
# @Description :
import os
import pandas as pd
import numpy as np

def loadData(csvFile):
    pickleDump = '{}DroppedNaNCols.pickle'.format(csvFile)
    if os.path.exists(pickleDump):
        df = pd.read_pickle(pickleDump)
    else:
        df = pd.read_csv(csvFile, low_memory=False, na_values='NaN')
        # print(np.inf)
        # print(np.nan)
        # df.replace([np.inf, -np.inf], np.nan)
        # clean data
        # strip the whitspaces from column names
        df = df.rename(str.strip, axis='columns')
        # drop Infinity rows and NaN string from each column
        for col in df.columns:
            test = df[col]
            test2 = test == np.inf
            # print('true_index_list', test2.index[test2])
            # true_index_list = [i for i in test2.index if test2[i]]
            # if not len(true_index_list):
            #     print('true_index_list:',true_index_list)
            # Using Boolean Indexing
            indexNames = df[df[col] == np.inf].index
            if not indexNames.empty:
                print('deleting {} rows with Infinity in column {}'.format(len(indexNames), col))
                df.drop(indexNames, inplace=True)

        df.argPathRatio = df['argPathRatio'].astype('float')
        # drop all columns with NaN values
        beforeColumns = df.shape[1]
        df.dropna(axis='columns', inplace=True)
        print('Dropped {} columns with NaN values'.format(beforeColumns - df.shape[1]))
        # drop all rows with NaN values
        beforeRows = df.shape[0]
        df.dropna(inplace=True)
        print('Dropped {} rows with NaN values'.format(beforeRows - df.shape[0]))
        df.to_pickle(pickleDump)

    return df

df = loadData('FinalDataset/ALL.csv')