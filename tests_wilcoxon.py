'''
Created on Nov 5, 2017

@author: anton

Wilcoxon tests
'''

from scipy.stats import ranksums, wilcoxon

import numpy as np
import pandas as pd

CLF_SGM = ['mlp-sgm', 'rbf', 'rbf-reg', 'mlp-sgm-2', 'conv-sgm']
CLF_SFT = ['mlp-sft', 'mlp-sft-2', 'conv-sft']

CLF_MLP_SGM = ['mlp-sgm', 'mlp-sgm-2', 'conv-sgm']
CLF_MLP_SFT = ['mlp-sft', 'mlp-sft-2', 'conv-sft']

CLF_MLP = ['mlp-sgm', 'mlp-sgm-2', 'conv-sgm', 'mlp-sft', 'mlp-sft-2', 'conv-sft']

MIS_PATHS = [
    'tests/moons_overlapping/run_1507477646.csv',
    'tests/multiclass/run_1507479282.csv',
    'tests/glass/run_1507570241.csv',
    'tests/segmentation/run_1507733898.csv',
    'tests/mnist/run_1507890155.csv',
    'tests/alphanumeric/run_1507997994.csv'
    ]

OUT_PATHS = [
    'tests/moons_separable/run_1507495890.csv',
    'tests/multiclass/run_1507497846.csv',
    'tests/iris/run_1507568403.csv',
    'tests/segmentation/run_1507813954.csv',
    'tests/mnist/run_1507902063.csv',
    'tests/alphanumeric/run_1508061367.csv'
    ]


def wicoxon_test_1(filename, params):
    """
    Some old tests
    """
    df = pd.read_csv(filename)
    print(df.keys())
    
    mrt = [df.loc[(
        (df['Clf'] == param[1])
        & (df['Units'] == param[3])
        & (df['Beta'] == param[4])
        & (df['DO'] == param[5])
        & (df['ES'] == param[6])
        & (df['Trgt'] == str(param[7]))
        ), 'MDT'].values for param in params]
    
    print(mrt)
    
    A = [[ranksums(a, b) for a in mrt] for b in mrt]
    
    print('\n'.join([','.join(['{:4f} '.format(item) for item in row]) for row in A]))


def wilcoxon_test_2(filename):
    """
    Compare output and differential thresholds for misclassifiaction errors
    for networks exept with softmax output layer
    """
    df = pd.read_csv(filename)
    print(filename)
    
    values = df.loc[(df['Clf'].isin(CLF_SGM)), ['SOT', 'SDT']].values
    wr = wilcoxon(values[:, 0], values[:, 1])
    print(wr)
    
    values = df.loc[(df['Clf'].isin(CLF_SGM)), ['MOT', 'MDT']].values
    wr = wilcoxon(values[:, 0], values[:, 1])
    print(wr)
    
    print()


def wilcoxon_test_3(filename):
    """
    Compare softmax and sigmoid activation functions for the output layer
    """
    df = pd.read_csv(filename)
    print(filename)
    
    a = df.loc[(df['Clf'].isin(CLF_MLP_SGM)), ['SDT']].values
    b = df.loc[(df['Clf'].isin(CLF_MLP_SFT)), ['SDT']].values
    wr = ranksums(a, b)
    print(wr)
    
    a = df.loc[(df['Clf'].isin(CLF_MLP_SGM)), ['Tst acc']].values
    b = df.loc[(df['Clf'].isin(CLF_MLP_SFT)), ['Tst acc']].values
    wr = ranksums(a, b)
    print(wr)
    
    print()


def wilcoxon_test_4(filename, column):
    """
    Compare softmax and sigmoid activation functions for the output layer
    """

    def clf_map(df, ind):
        clf = df['Clf'].loc[ind]
        if clf in ['mlp-sgm', 'mlp-sft']: return 'mlp'
        elif clf in ['mlp-sgm-2', 'mlp-sft-2']: return 'mlp-2'
        elif clf in ['conv-sgm', 'conv-sft']: return 'conv'
        else: raise ValueError('Unknown Clf')
    
    df = pd.read_csv(filename)
    print(filename)
    
    values = df[(df['Clf'].isin(CLF_MLP))].groupby([lambda x: clf_map(df, x), 'Units', 'Beta', 'DO', 'ES', 'Trgt'])[column].apply(list).values
    values = np.array([np.array(x) for x in values])
    wr = wilcoxon(values[:, 0], values[:, 1])
    print(wr)
    
    print()


def test(filename):
    
    df = pd.read_csv(filename)
    
    print(filename)
    
    # print(df['Clf'])
    
    values = df.loc[(df['Clf'].isin(CLF_SGM)), ['Clf', 'MOT', 'MDT']]
    
    print(values)


if __name__ == '__main__':
    # for path in MIS_PATHS:
        # wilcoxon_test_2(path)
        # wilcoxon_test_4(path, 'SDT')
        # wilcoxon_test_4(path, 'Tst acc')
    
    # for path in OUT_PATHS:
        # wilcoxon_test_2(path)
        # wilcoxon_test_4(path, 'SOT')
        # wilcoxon_test_4(path, 'Tst acc')
    
    test(MIS_PATHS[0])
