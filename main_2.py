'''
Created on Nov 12, 2016

@author: anton
'''
from input_data import get_data
from graphics import draw_roc, draw_precision_recall, plot_confusion_matrix, plot_multiclass_roc_curve
from graphics import plot_multiclass_precision_recall_curve
from data_utils import calc_roc, calc_precision_recall
from MLP import MLP
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def test(ds, activation_function='softmax', optimizer='adagrad', learning_rate=0.1, steps=100001, batch_size=256, hidden_layer=[2], regularization_penalty=0.0, early_stopping=None, graphics=False):
    print('learning rate {:g}; hidden_num {:s}'.format(learning_rate, str(hidden_layer)))
        
    if early_stopping is None:
        trn_x, trn_y, tst_x, tst_y, class_names = ds
        vld_x = tst_x
        vld_y = tst_y
    else:
        trn_x, trn_y, vld_x, vld_y, tst_x, tst_y, _ = ds
        
    mlp = MLP(learning_rate, [trn_x.shape[1]]+hidden_layer+[trn_y.shape[1]], activation_function, optimizer, regularization_penalty)
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
        # plot confusion matrix
        cnf_matrix = confusion_matrix(y_true, predictions)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names)
        plt.show()
        
        #plot ROC curves for multiple output thresholds
        plt.figure()
        plot_multiclass_roc_curve(tst_y, outputs, class_names)
        plt.show()
        
        # plot ROC curve for different rejection methods
        fpr, tpr, roc_auc = calc_roc(y_true, outputs)
        plt.figure()
        draw_roc(fpr, tpr, roc_auc, 'tests/roc_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        plt.show()
        
        #plot Precision-Recall curves for multiple output threshols
        plt.figure()
        plot_multiclass_precision_recall_curve(tst_y, outputs, class_names)
        plt.show()
        
        # plot precision-recall curve
        precision, recall, average_precision = calc_precision_recall(y_true, outputs)
        plt.figure()
        draw_precision_recall(precision, recall, average_precision, 'tests/prr_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        
        plt.show()
    return ''


if __name__ == '__main__':
    ds_x, ds_y, class_names = get_data(1, binarize=True, preprocess=4)
    ds = train_test_split(ds_x, ds_y, test_size=0.5, random_state=42)
    ds = [ds[0], ds[2], ds[1], ds[3], class_names]
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
    #test(ds, 'sigmoid',     'adam', 0.01, 2000, 128, [20,20], graphics=False)
    #test(ds, 'sigmoid',     'adam', 0.01, 2000, 512, [20,20], graphics=False)
    #
    #test(ds, 'sigmoid',     'adam', 0.01, 2000, 128, [20,20], 0.5, graphics=False)
    #test(ds, 'sigmoid',     'adam', 0.01, 2000, 128, [20,20], 1.0, graphics=False)
    
    # one month cancel_dump_traceback_later
    
    test(ds, 'sigmoid', 'gradient', 0.1, 2000, 128, [32], graphics=True)