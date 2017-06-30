'''
Created on Jun 24, 2017

@author: anton
'''
from DataSet import DataSet
from MLP import MLP
import time
from RBF import RBF
from klnvch.rejection_option.core import RejectionOption, get_labels

def test_unit_RBF(ds, n_hidden):
    rbf = RBF(ds.n_features, n_hidden, ds.n_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    
    score = rbf.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    ro = RejectionOption(rbf, ds.n_classes)
    line = ro.evaluate(ds.tst.x, ds.tst.y, ds.outliers)
    
    return '{:9f}, {:s}'.format(score, line)

def test_unit_mlp_rc(ds, clf, noise_size, units, beta, dropout, es, show):
    ds = ds.copy()
    ds.add_noise_as_a_class(noise_size)
    
    if clf == 'mlp-softmax':
        mlp = MLP(0.01, [ds.n_features, units, ds.n_classes],
                  ['sigmoid', 'softmax'], 'Adam', beta, 40)
    elif clf == 'mlp-sigmoid':
        mlp = MLP(0.01, [ds.n_features, units, ds.n_classes],
                  ['sigmoid', 'sigmoid'], 'Adam', beta, 40)
    
    result = mlp.train(10000, ds.trn, ds.vld, dropout, es, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    ro = RejectionOption(mlp, ds.n_classes, rc=True)
    line = ro.evaluate(ds.tst.x, ds.tst.y, ds.outliers)
    
    return '{:9f}, {:9f}, {:9f}, {:9f}, {:s}' \
        ''.format(result[2], result[3], result[4], score, line)

def test_unit_mlp(ds, noise_size, units, beta, dropout, es, show):
    ds = ds.copy()
    ds.add_noise_as_no_class(noise_size)
    
    mlp = MLP(0.01, [ds.n_features, units, ds.n_classes],
              ['sigmoid', 'sigmoid'], 'Adam', beta, 40)
    result = mlp.train(10000, ds.trn, ds.vld, dropout, es, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    ro = RejectionOption(mlp, ds.n_classes)
    line = ro.evaluate(ds.tst.x, ds.tst.y, ds.outliers)
    if show:
        ro.plot()
        ro.print_thresholds()
    
    return '{:f}, {:f}, {:f}, {:f}, {:s}' \
        .format(result[2], result[3], result[4], score, line)

def test_block_RBF(ds_name, ds_id, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    colums = 'DS,Attempt,Units,Tst acc,' + get_labels(3)
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, split=[0.2, 0.8], add_noise=3)
        for param in params:
            n_hidden = param
            msg = test_unit_RBF(ds, n_hidden)
            msg = '{:s}, {:d}, {:d}, '.format(ds_name, attempt, n_hidden) + msg
            with open(filename, 'a+') as f:
                print(msg, file=f)

def test_block_mlp_rc(ds_id, ds_name, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    
    colums = 'DS,Clf,Attempt,Type,Noise,Units,Beta,Dropout,ES,Loss,Trn acc,Vld acc,Tst acc,' \
                     + get_labels(3, rc=True)
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, split=[0.3, 0.1, 0.6])
        for param in params:
            noise_type, clf, noise_size, units, beta, dropout, es, _ = param
            msg = test_unit_mlp_rc(ds, clf, noise_size, units, beta,
                                dropout, es, False)
            
            msg = '{:s}, {:s}, {:d}, {:d}, {:f}, {:d}, {:f}, {:f}, {:d}, ' \
                .format(ds_name, clf, attempt, noise_type, noise_size, units, beta, dropout, es) + msg
            
            with open(filename, 'a+') as f:
                print(msg, file=f)

def test_block_mlp(ds_id, ds_name, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    
    colums = 'DS,Clf,Attempt,Type,Noise,Units,Beta,Dropout,ES,Loss,Trn acc,Vld acc,Tst acc,' \
                     + get_labels(3)
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, split=[0.3, 0.1, 0.6])
        for param in params:
            noise_type, clf, noise_size, units, beta, dropout, es, _ = param
            msg = test_unit_mlp(ds, noise_size, units, beta,
                                dropout, es, False)
            
            msg = '{:s}, {:s}, {:d}, {:d}, {:f}, {:d}, {:f}, {:f}, {:d}, ' \
                .format(ds_name, clf, attempt, noise_type, noise_size, units, beta, dropout, es) + msg
            
            with open(filename, 'a+') as f:
                print(msg, file=f)

if __name__ == '__main__':
    ds = DataSet(2, split=[0.3, 0.1, 0.6])
    test_unit_mlp(ds, 4.0, 16, 0.0001, 1.0, 0, True)
    """
    test_block_mlp(2, 'iris', range(0,1),
                   [[1, 'mlp-sigmoid', 0.5, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 1.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 2.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 8.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0,  4, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0,  8, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 12, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.001,   1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.00001, 1.0, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.0001,  0.9, 0,  (0.0, 1.0)],
                    [1, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 50, (0.0, 1.0)]])
    """
    """
    test_block_mlp_rc(2, 'iris', range(0,1),
                   [[2, 'mlp-sigmoid', 0.5, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 1.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 2.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 8.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0,  4, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0,  8, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 12, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.001,   1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.00001, 1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.0001,  0.9, 0,  (0.0, 1.0)],
                    [2, 'mlp-sigmoid', 4.0, 16, 0.0001,  1.0, 50, (0.0, 1.0)],
                    [2, 'mlp-softmax', 0.5, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 1.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 2.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 8.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0,  4, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0,  8, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 12, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.001,   1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.0001,  1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.00001, 1.0, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.0001,  0.9, 0,  (0.0, 1.0)],
                    [2, 'mlp-softmax', 4.0, 16, 0.0001,  1.0, 50, (0.0, 1.0)]])
    """
    """
    test_block_RBF('iris', 2, range(0,1),[3,6,9,12,15,18,21,24])
    """