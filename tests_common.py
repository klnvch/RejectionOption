'''
Created on Aug 24, 2017

@author: anton

Some common code between tests
'''
import csv
import time

from scipy.stats.stats import ranksums

from DataSet import DataSet
from MLP import MLP
from RBF import RBF
from klnvch.rejection_option.core import RejectionOption
import numpy as np
import pandas as pd


# N_EPOCHS = 5000
# BATCH_SIZE = 128
LEARNING_RATE = 0.01

def test_unit_mlp(ds, clf, rej, units, beta, dropout, es, targets, n_epochs,
                  batch_size, print_step, show, path=None, suffix=None):
    ds = ds.copy()
    ds.add_outliers(rej)
    ds.change_targets(targets)
    
    if clf == 'mlp-sft':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['sigmoid', 'softmax'], 'Adam', beta, batch_size)
    if clf == 'mlp-sft-2':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, units, ds.n_classes],
                  ['sigmoid', 'sigmoid', 'softmax'], 'Adam', beta, batch_size)
    elif clf == 'mlp-sgm':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['sigmoid', 'sigmoid'], 'Adam', beta, batch_size)
    elif clf == 'mlp-sgm-2':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, units, ds.n_classes],
                  ['sigmoid', 'sigmoid', 'sigmoid'], 'Adam', beta, batch_size)
    elif clf == 'mlp-relu':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, units, ds.n_classes],
                  ['relu', 'relu', 'softmax'], 'Adam', beta, batch_size)
    elif clf == 'rbf':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['rbf', 'sigmoid'], 'Adam', beta, batch_size)
    elif clf == 'rbf-reg':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['rbf-reg', 'sigmoid'], 'Adam', beta, batch_size)
    elif clf == 'conv-sgm':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['conv', 'relu', 'sigmoid'], 'Adam', beta, batch_size)
    elif clf == 'conv-sft':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['conv', 'relu', 'softmax'], 'Adam', beta, batch_size)
    
    result = mlp.train(n_epochs, ds.trn, ds.vld, dropout, es, print_step, False)
    # result = [0,0,0,0,0,0,0,0,0,0]
    print(result)
    
    rc = rej not in [0, 1, 3, 5, 8]
    
    ro = RejectionOption(mlp, ds.target_names,
                         ds.tst.x, ds.tst.y, ds.outliers, rc)
    ro.set_verbosity(show, path, suffix)
    ro.print_classification_report()
    metrics = ro.calc_metrics()
    ro.plot_decision_regions(mlp)
    ro.plot_confusion_matrix()
    ro.plot_roc()
    ro.plot_roc_precision_recall()
    ro.plot_multiclass_roc()
    ro.plot_multiclass_precision_recall()
    
    _, step, trn_loss, trn_acc, vld_acc = result
    return np.concatenate(([step, trn_loss, trn_acc, vld_acc], metrics))

def test_block_mlp(ds_id, ds_name, attempts, n_samples, split, params,
                   random_state=None):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    fig_path = 'tests/{:s}/figures/'.format(ds_name)
    
    columns = 'DS,Clf,Attempt,Type,Units,Beta,DO,ES,Trgt,Epochs,' \
                'Loss,Trn acc,Vld acc,Tst acc,SOT,SDT,SRT,MOT,MDT,MRT'
    
    with open(filename, 'a+') as f:
        print(columns, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, n_samples=n_samples, split=split,
                     random_state=random_state)
        for param in params:
            fig_suff = '_'.join(map(str, param))
            rej, clf, units, beta, dropout, es, targets, \
                n_epochs, batch_size, print_step = param
            
            msg = test_unit_mlp(ds, clf, rej, units, beta, dropout, es, targets,
                                n_epochs, batch_size, print_step, False,
                                fig_path, fig_suff)
            
            if es == 0: es = '-'
            else:       es = '+'
            if targets == (0.0, 1.0):   targets = '-'
            else:                       targets = '+'
            row = np.concatenate(([ds_name, clf, attempt, rej, units, beta,
                                   dropout, es, targets], msg))
            
            with open(filename, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(row)

def test_unit_RBF(ds, n_hidden, beta=None, show=False, path=None, suffix=None):
    rbf = RBF(ds.n_features, n_hidden, ds.n_classes, beta)
    rbf.train(ds.trn.x, ds.trn.y)
    
    trn_score = rbf.score(ds.trn)
    # guess = rbf.predict_proba([np.random.rand(153)])[0]
    # print('Random output: {:d}'.format(guess.argmax()))
    print('Train accuracy: {0:f}'.format(trn_score))
    
    ro = RejectionOption(rbf, ds.target_names,
                         ds.tst.x, ds.tst.y, ds.outliers, False)
    ro.set_verbosity(show, path, suffix)
    ro.print_classification_report()
    metrics = ro.calc_metrics()
    ro.plot_decision_regions(rbf)
    ro.plot_confusion_matrix()
    ro.plot_roc()
    ro.plot_roc_precision_recall()
    ro.plot_multiclass_roc()
    ro.plot_multiclass_precision_recall()
    
    return np.concatenate(([trn_score], metrics))

def test_block_rbf(ds_id, ds_name, attempts, n_samples, split, params,
                   random_state=None):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    fig_path = 'tests/{:s}/figures/'.format(ds_name)
    
    colums = 'DS,Clf,Attempt,Units,Beta,Trn acc,Tst acc,SOT,SDT,SRT,MOT,MDT,MRT'
    
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, n_samples=n_samples, split=split,
                     random_state=random_state)
        ds.add_outliers(3)
        for param in params:
            distance, n_hidden = param
            fig_suff = '3_rbf-km_'.join(map(str, param))
            
            msg = test_unit_RBF(ds, n_hidden, distance,
                                False, fig_path, fig_suff)
            
            row = np.concatenate(([ds_name, 'rbf-km', attempt, n_hidden,
                                   distance], msg))
            
            with open(filename, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(row)

def wicoxon_test(filename, params):
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
