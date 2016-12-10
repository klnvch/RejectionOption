'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import heapq
from collections import Counter
from sklearn import metrics

def count_distribution(y):
    d = [0] * y.shape[1]
    
    for i in y:
        d[i.argmax()] += 1
        
    #d = np.asarray(d) / ds_y.shape[0]
    print(d)
    return d

def print_frequencies(x):
    print(Counter(x))

def remove_class(ds_x, ds_y, i):
    new_ds_x = []
    new_ds_y = []
    outliers = []
    
    for x, y in zip(ds_x, ds_y):
        if y.argmax() in i:
            outliers.append(x)
        else:
            new_ds_x.append(x)
            new_ds_y.append(np.delete(y, i))
    
    return np.array(new_ds_x), np.array(new_ds_y), np.array(outliers)

def add_noise_as_no_class(ds_x, ds_y, noise_size=None, noise_output=None):
    assert ds_x.shape[0] == ds_y.shape[0]
    
    if noise_size is None:
        noise_size = ds_y.shape[0]
        
    if noise_output is None:
        noise_output = 1.0/ds_y.shape[1]
    
    noise = np.random.uniform(ds_x.min(), ds_x.max(), [noise_size, ds_x.shape[1]])
    
    new_ds_x = np.concatenate([ds_x, noise])
    new_ds_y = np.concatenate([ds_y, np.array([[noise_output]*ds_y.shape[1]] * noise_size)])
    
    assert new_ds_x.shape[0] == new_ds_y.shape[0]
    return new_ds_x, new_ds_y

def add_noise_as_a_class(ds_x, ds_y, noise_size=None):
    assert ds_x.shape[0] == ds_y.shape[0]
    
    if noise_size is None:
        noise_size = ds_y.shape[0]
    
    noise = np.random.uniform(0.0, 1.0, [noise_size, ds_x.shape[1]])
    
    ds_y = np.append(ds_y, np.array([[0]] * ds_y.shape[0]), axis=1)
    
    new_ds_x = np.concatenate([ds_x, noise])
    new_ds_y = np.concatenate([ds_y, np.array([[0]*ds_y.shape[1] + [1]] * noise_size)])
    
    assert new_ds_x.shape[0] == new_ds_y.shape[0]
    return new_ds_x, new_ds_y

def rejection_score(outputs, rejection_method):
    if rejection_method == 0:
        return np.max(outputs, axis=1)
    elif rejection_method == 1:
        result = []
        for o in outputs:
            x = heapq.nlargest(2, o)
            result.append(x[0] - x[1])
        return result
    elif rejection_method == 2:
        result = []
        for o in outputs:
            x = heapq.nlargest(2, o)
            result.append(1.0 - x[1] / x[0])
        return result
    else:
        assert False
        
def calc_roc(y, outputs):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    predictions = [np.argmax(o) for o in outputs]
    #y_true = np.array(y) - np.array(predictions)
    y_true = [a==b for a,b in zip(np.array(y),np.array(predictions))]
    
    for i in [0,1,2]:
        y_score = rejection_score(outputs, i)
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score)
        #roc_auc[i] = metrics.roc_auc_score(y_true, y_score)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    return fpr, tpr, roc_auc

def calc_precision_recall(y, outputs):
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    predictions = [np.argmax(o) for o in outputs]
    #y_true = np.array(y) - np.array(predictions)
    y_true = [a==b for a,b in zip(np.array(y),np.array(predictions))]
    
    for i in [0,1,2]:
        y_score = rejection_score(outputs, i)
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true, y_score)
        average_precision[i] = metrics.average_precision_score(y_true, y_score)
        
    return precision, recall, average_precision