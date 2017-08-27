'''
Created on Aug 24, 2017

@author: anton

Some common code between tests
'''
import csv
import numpy as np
import time

from DataSet import DataSet
from MLP import MLP
from RBF import RBF
from klnvch.rejection_option.core import RejectionOption

#N_EPOCHS = 5000
BATCH_SIZE = 32
LEARNING_RATE = 0.01

def test_unit_mlp(ds, clf, rej, units,
                  beta, dropout, es, targets, n_epochs, show):
    ds = ds.copy()
    ds.change_targets(targets)
    
    if clf == 'mlp-sft':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['sigmoid', 'softmax'], 'Adam', beta, BATCH_SIZE)
    if clf == 'mlp-sft-2':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, units, ds.n_classes],
                  ['sigmoid', 'sigmoid', 'softmax'], 'Adam', beta, BATCH_SIZE)
    elif clf == 'mlp-sgm':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['sigmoid', 'sigmoid'], 'Adam', beta, BATCH_SIZE)
    elif clf == 'mlp-sgm-2':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, units, ds.n_classes],
                  ['sigmoid', 'sigmoid', 'sigmoid'], 'Adam', beta, BATCH_SIZE)
    elif clf == 'mlp-relu':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, units, ds.n_classes],
                  ['relu', 'relu', 'softmax'], 'Adam', beta, BATCH_SIZE)
    elif clf == 'rbf':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['rbf', 'sigmoid'], 'Adam', beta, BATCH_SIZE)
    elif clf == 'rbf-reg':
        mlp = MLP(LEARNING_RATE, [ds.n_features, units, ds.n_classes],
                  ['rbf-reg', 'softmax'], 'Adam', beta, BATCH_SIZE)
    
    result = mlp.train(n_epochs, ds.trn, ds.vld, dropout, es, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    if rej in [0,1,3]:
        ro = RejectionOption(mlp, ds.n_classes, False, 'simple')
    elif rej == 2:
        ro = RejectionOption(mlp, ds.n_classes, True, 'simple')
    
    ro.init(ds.target_names, ds.tst.x, ds.tst.y, ds.outliers)
    ro.print_classification_report()
    line = ro.calc_metrics()
    if show:
        ro.plot_decision_regions()
        ro.plot_confusion_matrix()
        ro.plot_roc()
        ro.plot_roc_precision_recall()
        ro.plot_multiclass_roc()
        ro.plot_multiclass_precision_recall()
    
    return np.concatenate(([result[2], result[3], result[4], score], line))

def test_block_mlp(ds_id, ds_name, rej, attempts, size, split, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    
    columns = 'DS,Clf,Epochs,Attempt,Type,Units,Beta,Dropout,ES,' \
                'Loss,Trn acc,Vld acc,Tst acc,SOT,SDT,SRT,MOT,MDT,MRT'
    
    with open(filename, 'a+') as f:
        print(columns, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, size=size, split=split, add_noise=rej)
        for param in params:
            clf, units, beta, dropout, es, targets, \
                n_epochs = param
            
            msg = test_unit_mlp(ds, clf, rej, units,
                                beta, dropout, es, targets, n_epochs, False)
            
            row = np.concatenate(([ds_name, clf, n_epochs, attempt, rej,
                                   units, beta, dropout, es], msg))
            
            with open(filename, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(row)

def test_unit_RBF(ds, n_hidden, beta=None, show=False):
    rbf = RBF(ds.n_features, n_hidden, ds.n_classes, beta)
    rbf.train(ds.trn.x, ds.trn.y)
    
    trn_score = rbf.score(ds.trn)
    tst_score = rbf.score(ds.tst)
    #guess = rbf.predict_proba([np.random.rand(153)])[0]
    #print('Random output: {:d}'.format(guess.argmax()))
    print('Train accuracy: {0:f}'.format(trn_score))
    print('Test accuracy: {0:f}'.format(tst_score))
    
    ro = RejectionOption(rbf, ds.n_classes, False, 'simple')
    ro.init(ds.target_names, ds.tst.x, ds.tst.y, ds.outliers)
    ro.print_classification_report()
    line = ro.calc_metrics()
    if show:
        ro.plot_decision_regions()
        ro.plot_confusion_matrix()
        ro.plot_roc()
        ro.plot_roc_precision_recall()
        ro.plot_multiclass_roc()
        ro.plot_multiclass_precision_recall()
    
    return np.concatenate(([trn_score, tst_score], line))

def test_block_rbf(ds_name, ds_id, attempts, size, split, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    colums = 'DS,Attempt,Units,Beta,Tst acc,'
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, size=size, split=split, add_noise=0)
        for param in params:
            distance, n_hidden = param
            msg = test_unit_RBF(ds, n_hidden, distance)
            
            row = np.concatenate(([ds_name, attempt, n_hidden, distance], msg))
            
            with open(filename, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(row)