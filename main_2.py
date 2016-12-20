'''
Created on Nov 12, 2016

@author: anton
'''
from input_data import get_data
from graphics import draw_roc, draw_precision_recall
from data_utils import calc_roc, calc_precision_recall
from MLP import MLP
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics

def test(ds, activation_function='softmax', optimizer='adagrad', learning_rate=0.1, steps=100001, batch_size=256, hidden_layer=[2], early_stopping=None, graphics=False):
    print('learning rate {:g}; hidden_num {:s}'.format(learning_rate, str(hidden_layer)))
        
    if early_stopping is None:
        trn_x, trn_y, tst_x, tst_y, _ = ds
        vld_x = tst_x
        vld_y = tst_y
    else:
        trn_x, trn_y, vld_x, vld_y, tst_x, tst_y, _ = ds
        
    mlp = MLP(learning_rate, [trn_x.shape[1]]+hidden_layer+[trn_y.shape[1]], activation_function, optimizer)
    result = mlp.train(steps, trn_x, trn_y, vld_x, vld_y, batch_size, early_stopping=early_stopping, logging=True)
    
    if early_stopping is None:
        print(result)
        
        outputs = mlp.predict_proba(tst_x, result[0])
        predictions = [np.argmax(o) for o in outputs]
        y_true = [np.argmax(y) for y in tst_y]
        score = metrics.accuracy_score(y_true, predictions)
        print('Accuracy: {0:f}'.format(score))
        report = metrics.classification_report(y_true, predictions)
        print(report)
        
    else:
        print(result[0])
        print(result[1])
        print(result[2])
        print(result[3])
        
        tst_acc_0 = mlp.test(tst_x, tst_y, result[0][0], logging=True)
        tst_acc_1 = mlp.test(tst_x, tst_y, result[1][0], logging=True)
        tst_acc_2 = mlp.test(tst_x, tst_y, result[2][0], logging=True)
        tst_acc_3 = mlp.test(tst_x, tst_y, result[3][0], logging=True)
        print([tst_acc_0, tst_acc_1, tst_acc_2, tst_acc_3])
    
    if graphics:
        fpr, tpr, roc_auc = calc_roc(y_true, outputs)
        draw_roc(fpr, tpr, roc_auc, 'tests/roc_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        #
        precision, recall, average_precision = calc_precision_recall(y_true, outputs)
        draw_precision_recall(precision, recall, average_precision, 'tests/prr_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        #
    return ''


if __name__ == '__main__':
    ds_x, ds_y, _ = get_data(1, binarize=True, preprocess=4)
    ds = train_test_split(ds_x, ds_y, test_size=0.5, random_state=42)
    ds = [ds[0], ds[2], ds[1], ds[3], None]
    #
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [28], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [32], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [36], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [40], graphics=False)
    #
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [28], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [32], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [36], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [40], graphics=False)
    #
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [28], graphics=False)
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [32], graphics=False)
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [36], graphics=False)
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [40], graphics=False)
    #
    #test(ds, 'sigmoid', 'gradient', 0.01, 20000, 256, [20,20], graphics=False)
    #test(ds, 'sigmoid',     'adam', 0.01, 20000, 256, [20,20], graphics=False)
    #test(ds, 'sigmoid',  'adagrad', 0.01, 20000, 256, [20,20], graphics=False)
    #
    test(ds, 'sigmoid',     'adam', 0.01, 2000, 128, [20,20], graphics=False)
    test(ds, 'sigmoid',     'adam', 0.01, 2000, 512, [20,20], graphics=False)