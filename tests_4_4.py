'''
Created on Jun 24, 2017

@author: anton
'''
import csv
import numpy as np
import pandas as pd
from scipy.stats import ranksums
import time
from DataSet import DataSet
from RBF import RBF
from klnvch.rejection_option.core import RejectionOption
from tests_common import test_unit_mlp

def test_unit_RBF(ds, n_hidden, beta=None, show=False):
    rbf = RBF(ds.n_features, n_hidden, ds.n_classes, beta)
    rbf.train(ds.trn.x, ds.trn.y)
    
    trn_score = rbf.score(ds.trn)
    tst_score = rbf.score(ds.tst)
    guess = rbf.predict_proba([np.random.rand(153)])[0]
    print('Random output: {:d}'.format(guess.argmax()))
    print('Train accuracy: {0:f}'.format(trn_score))
    print('Test accuracy: {0:f}'.format(tst_score))
    
    ro = RejectionOption(rbf, ds.n_classes, False, 'simple')
    ro.init(ds.target_names, ds.tst.x, ds.tst.y, ds.outliers)
    ro.print_classification_report()
    line = ro.calc_metrics()
    if show:
        ro.plot_confusion_matrix()
        ro.plot_roc()
        ro.plot_roc_precision_recall()
        ro.plot_multiclass_roc()
        ro.plot_multiclass_precision_recall()
    
    return np.concatenate(([trn_score, tst_score], line))

def test_block_rbf(ds_name, ds_id, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    colums = 'DS,Attempt,Units,Beta,Tst acc,'
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, add_noise=3)
        for param in params:
            distance, n_hidden = param
            msg = test_unit_RBF(ds, n_hidden, distance)
            
            row = np.concatenate(([ds_name, attempt, n_hidden, distance], msg))
            
            with open(filename, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(row)

# rej, clf, noisi_size, units, beta, dropout, es, targets, epochs
params_0 = [
    ['mlp-sigmoid', 16,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 32,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 256, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.0001,   1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.000001, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.8, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.5, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.5, 10, (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.5, 0,  (0.1, 0.9), 5000],
    
    ['mlp-softmax', 16,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 32,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 256, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.0001,   1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.000001, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.00001,  0.8, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.00001,  0.5, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.00001,  0.5, 10, (0.0, 1.0), 5000]
    ]

params_1 = [
    ['mlp-sigmoid', 16,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 32,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64,  0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 256, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.0001,   1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.000001, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.8, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.5, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.5, 10, (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  0.5, 0,  (0.1, 0.9), 5000],
    ]

params_2 = [
    ['mlp-sigmoid', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid',  8, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 16, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 32, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.0001,   1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.000001, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  0.8, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 64, 0.00001,  1.0, 50, (0.0, 1.0), 5000],
    
    ['mlp-softmax', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax',  8, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 16, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 32, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.0001,   1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.000001, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  0.8, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 64, 0.00001,  1.0, 50, (0.0, 1.0), 5000],
    ]

params_4 = [
    [0.000001, 32],
    [0.000001, 64],
    [0.000001, 128],
    [0.000001, 256],
    [0.000001, 512],
    
    [0.00001, 32],
    [0.00001, 64],
    [0.00001, 128],
    [0.00001, 256],
    [0.00001, 512],
    
    [0.0001, 32],
    [0.0001, 64],
    [0.0001, 128],
    [0.0001, 256],
    [0.0001, 512],
    
    [0.001, 32],
    [0.001, 64],
    [0.001, 128],
    [0.001, 256],
    [0.001, 512],
    
    [0.01, 32],
    [0.01, 64],
    [0.01, 128],
    [0.01, 256],
    [0.01, 512]
    ]

def wicoxon_test(filename, params):
    df = pd.read_csv(filename)
    print(df.keys())

    mrt = [df.loc[(
        (df['Clf'] == param[1])
        & (df['Units']   == param[3])
        & (df['Beta']    == param[4])
        & (df['Dropout'] == param[5])
        & (df['ES']      == param[6])
        & (df['Targets'] == str(param[7]))
        ), 'MDT'].values for param in params]
    
    print(mrt)
    
    A = [[ranksums(a, b).statistic for a in mrt] for b in mrt]
    
    print('\n'.join([','.join(['{:4f} '.format(item) for item in row]) for row in A]))

if __name__ == '__main__':
    #wicoxon_test('tests/alphanumeric/run_0.csv', params_0)
    #wicoxon_test('tests/alphanumeric/run_1.csv', params_1)
    
    #ds = DataSet(13, add_noise=0)#, output=(0.1, 0.9))
    #test_unit_mlp(ds, 'mlp-sigmoid', 0, None, 128, 0.00001, 0.5, 0, (0.0, 1.0), 5000, True)
    
    #ds = DataSet(13, add_noise=3)
    #test_unit_mlp(ds, 'rbf', 0, None, 128, 0.0, 1.0, 0, (0.0, 1.0), 5000, True)
    
    ds = DataSet(13, add_noise=3)
    test_unit_mlp(ds, 'rbf-reg', 0, None, 128, 0.0, 1.0, 0, (0.0, 1.0), 5000, True)
    
    #ds = DataSet(13, add_noise=3)
    #test_unit_RBF(ds, 128, 0.0001, True)
    
    #test_block_mlp(13, 'alphanumeric', 0, range(1,9), params_0)
    #test_block_mlp(13, 'alphanumeric', 1, range(9,10), params_1)
    #test_block_mlp(13, 'alphanumeric', 2, range(0,1), params_2)
    #test_block_rbf('alphanumeric', 13, range(0,1), params_4)