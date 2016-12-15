'''
Created on Nov 12, 2016

@author: anton
'''
from input_data import get_data
from data_utils import remove_class
from data_utils import add_noise_as_no_class
from data_utils import add_noise_as_a_class
from graphics import draw_roc, draw_precision_recall
from data_utils import calc_roc, calc_precision_recall
from MLP import MLP
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics

def test(ds, learning_rate, num_steps=100001, hidden_layer=[2], activation_function='softmax', early_stopping=50, graphics=False):
    print('learning rate {:f}; hidden_num {:s}'.format(learning_rate, str(hidden_layer)))
        
    if early_stopping is None:
        trn_x, trn_y, tst_x, tst_y, outliers = ds
        vld_x = tst_x
        vld_y = tst_y
    else:
        trn_x, trn_y, vld_x, vld_y, tst_x, tst_y, outliers = ds
        
    mlp = MLP(learning_rate, [trn_x.shape[1]]+hidden_layer+[trn_y.shape[1]], activation_function, 'adagrad')
    result = mlp.train(num_steps, trn_x, trn_y, vld_x, vld_y, batch_size=256, early_stopping=early_stopping, logging=True)
    
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
    
        c1 = mlp.test_rejection(tst_x, tst_y, outliers, 0, 100, result[1][0])
        c2 = mlp.test_rejection(tst_x, tst_y, outliers, 1, 100, result[2][0])
        c3 = mlp.test_rejection(tst_x, tst_y, outliers, 2, 100, result[3][0])
    
    if graphics:
        fpr, tpr, roc_auc = calc_roc(y_true, outputs)
        draw_roc(fpr, tpr, roc_auc, 'tests/roc_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        #
        precision, recall, average_precision = calc_precision_recall(y_true, outputs)
        draw_precision_recall(precision, recall, average_precision, 'tests/prr_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        #
    return ''

def generate_table_hidden_size(learning_rate, num_steps=100001, activation_function='softmax', hidden_layers=[[2]], early_stopping=None, add_noise=None, graphics=False):
    ds_x, ds_y, _ = get_data(1, binarize=True, preprocess=1)
    
    for attempt in [1]:
        if add_noise==1:
            #ds_x, ds_y, outliers = remove_class(ds_x, ds_y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            ds_x, ds_y, outliers = remove_class(ds_x, ds_y, [0])
        else:
            outliers=None

        if early_stopping is None:
            ds = train_test_split(ds_x, ds_y, test_size=0.5, random_state=42)
            if add_noise== 1:
                if activation_function=='softmax':
                    ds[0], ds[2] = add_noise_as_no_class(ds[0], ds[2], 500, 1.0/ds_y.shape[1])
                else:
                    ds[0], ds[2] = add_noise_as_no_class(ds[0], ds[2], 500, 0.0)
            ds = [ds[0], ds[2], ds[1], ds[3], outliers]
        else:
            ds1 = train_test_split(ds_x, ds_y, test_size=0.5)
            ds2 = train_test_split(ds1[1], ds1[3], test_size=0.5)
            ds = [ds1[0], ds1[2], ds2[0], ds2[2], ds2[1], ds2[3], outliers]
            
        results=[]
            
        for hidden_layer in hidden_layers:
            print('attempt: {:d}, hidden layer: {:s}'.format(attempt, str(hidden_layer)))
            result = test(ds, learning_rate, num_steps, hidden_layer, activation_function, early_stopping, graphics)
            results.append(result)
            print(result)
            
        with open('test.txt', 'a+') as log_file:
            print('-------------------------------------------------------------------------------------------', file=log_file)
            print('activation function: {:s}, early_stopping:'.format(activation_function), file=log_file)
            print('-------------------------------------------------------------------------------------------', file=log_file)
            for r in results:
                print('\shortstack{{{:s}}} & '.format(r), file=log_file)
            print('\hline', file=log_file)
            print('-------------------------------------------------------------------------------------------', file=log_file)

            

#generate_table_hidden_size(0.1, 20001, 'sigmoid', None, None, False)
generate_table_hidden_size(0.1, 10001, 'sigmoid', [[16,16]], None, None, False)
