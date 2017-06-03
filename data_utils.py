'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import heapq
from sklearn import metrics
from scipy import interp
import itertools

def count_distribution(y):
    """
    Check if needed
    """
    d = [0] * y.shape[1]
    
    for i in y:
        d[i.argmax()] += 1
        
    #d = np.asarray(d) / ds_y.shape[0]
    print(d)
    return d
    
def remove_class(x, y, names, indices):
    """
    Removes classes from indices in dataset (x, y)
    Args:
        x: features
        y: binarized outputs
        names: names of classes
        indices: list of classes to be removed
    Returns:
        new dataset, ds_x,ds_y and outliers
    """
    new_x = []
    new_y = []
    new_names = np.delete(names, indices)
    outliers = []
    for _x, _y in zip(x, y):
        if _y.argmax() in indices:
            outliers.append(_x)
        else:
            new_x.append(x)
            new_y.append(np.delete(_y, indices))
    return np.array(new_x), np.array(new_y), new_names, np.array(outliers)

def add_noise_as_no_class(x, y, noise_size=None, noise_output=None):
    """
    Adds noise with low output to a dataset
    Args:
        x: features
        y: binarized outputs
        noise_size: numner of noise patterns, default is side of the dataset
        noise_output: noise output, defult is 1./number of classes
    Returns:
        new dataset, ds_x,ds_y and outliers
    """
    assert x.shape[0] == y.shape[0]
    
    size = x.shape[0]    # number of patterns
    num_features = x.shape[1]  # number of features
    num_classes = y.shape[1]  # number of classes
    
    if noise_size is None: noise_size = size
    
    if noise_output is None: noise_output = 1.0/num_classes
    
    noise_x = np.random.uniform(x.min(), x.max(), [noise_size, num_features])
    noise_y = np.array([[noise_output] * num_classes] * noise_size)
    
    new_x = np.concatenate([x, noise_x])
    new_y = np.concatenate([y, noise_y])
    
    assert new_x.shape[0] == new_y.shape[0]
    return new_x, new_y

def add_noise_as_a_class(x, y, names, out_x=None, outliers_size=None):
    """
    Adds noise as a class to a dataset
    Args:
        x: features
        y: binarized outputs
        names: names of classes
        outliers: pattern, uniform distributed if None
        outliers_size: numner of noise patterns, default is side of the dataset
    Returns:
        new dataset, ds_x,ds_y and outliers
    """
    assert x.shape[0] == y.shape[0]
    
    size = x.shape[0]    # number of patterns
    num_features = x.shape[1]  # number of features
    num_classes = y.shape[1]  # number of classes
    
    if out_x is None:
        if outliers_size is None: outliers_size = size
        out_x = np.random.uniform(x.min(), x.max(), 
                                  [outliers_size, num_features])
    else:
        outliers_size = out_x.shape[0]
    
    out_y = np.array([[0]*(num_classes) + [1]] * outliers_size)
    
    new_x = np.concatenate([x, out_x])
    new_y = np.append(y, np.array([[0]] * size), axis=1) # add column
    new_y = np.concatenate([new_y, out_y])
    new_names = np.concatenate([names, ['Outliers']])
    
    assert new_x.shape[0] == new_y.shape[0]
    assert new_x.shape[0] == size + outliers_size
    
    return new_x, new_y, new_names

def rejection_score(outputs, rejection_method):
    """
    Compute scores for single threshold
    Args:
        outputs: real outputs
        rejection_method: 0-output, 1-differential, 2-ratio
    Returns:
        scores according to outputs
    """
    if   rejection_method == 0: return threshold_output(outputs)
    elif rejection_method == 1: return threshold_differential(outputs)
    elif rejection_method == 2: return threshold_ratio(outputs)
    else: raise ValueError('rejection method is wrong: ' + rejection_method)

def threshold_output(outputs):
    return np.max(outputs, axis=1)

def threshold_differential(outputs):
    result = []
    for o in outputs:
        x = heapq.nlargest(2, o)
        result.append(x[0] - x[1])
    return np.array(result)

def threshold_ratio(outputs):
    result = []
    for o in outputs:
        x = heapq.nlargest(2, o)
        result.append(1.0 - x[1] / x[0])
    return np.array(result)

def calc_roc_binary(outputs_true, outputs_pred, outputs_outliers=None):
    """
    Calcs binary ROC curve or for single threshold
    Args:
        outputs_true : desired output
        outputs_pred : real output
        outputs_outliers : output for outliers
    Returns:
        FPR, TPR, area under the ROC curve
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    
    for i in [0, 1, 2]: # across rejection methods
        if outputs_outliers is None:
            y_score = rejection_score(outputs_pred, i)
            fpr[i], tpr[i], _ = my_roc_curve(y_true, y_score)
        else:
            outputs = np.concatenate([outputs_pred, outputs_outliers])
            y_score = rejection_score(outputs, i)
            y_true_all = y_true + [False] * outputs_outliers.shape[0]
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true_all, y_score)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    return fpr, tpr, roc_auc

def my_roc_curve(y_true, y_score):
    fpr = []
    tpr = []
    
    for t in y_score:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for corectness, score in zip(y_true, y_score):
            if corectness:
                if score >= t: tp += 1
                else:          fn += 1
            else:
                if score >= t: fp += 1
                else:          tn += 1
        
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
    fpr, tpr, y_score = zip(*sorted(zip(fpr, tpr, y_score)))
    
    return fpr, tpr, y_score

def calc_roc_multiple(outputs_true, outputs_pred, labels):
    n_classes = len(labels)
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    y_score = threshold_output(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes)]
    y_score_classes = [[] for _ in range(n_classes)]
    for i, correctness, score in zip(y_ideal, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    
    fpr = []
    tpr = []
    
    for t0, t1 in itertools.product(y_score_classes[0], y_score_classes[1]):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        for corectness, score in zip(y_true_classes[0], y_score_classes[0]):
            if corectness:
                if score >= t0: tp += 1
                else:           fn += 1
            else:
                if score >= t0: fp += 1
                else:           tn += 1
        
        for corectness, score in zip(y_true_classes[1], y_score_classes[1]):
            if corectness:
                if score >= t1: tp += 1
                else:           fn += 1
            else:
                if score >= t1: fp += 1
                else:           tn += 1
        
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    
    clean_fpr = [fpr[0]]
    clean_tpr = [tpr[0]]
    for x, y in zip(fpr, tpr):
        if clean_fpr[-1] == x:
            clean_tpr[-1] = max(clean_tpr[-1], y) 
        else:
            clean_fpr.append(x)
            clean_tpr.append(y)
        
    
    return clean_fpr, clean_tpr, metrics.auc(clean_fpr, clean_tpr)

def calc_roc_multiclass(outputs_true, outputs_pred, labels,
                        outputs_outliers=None):
    """
    Calcs binary ROC curve or for multiple output thresholds
    Args:
        outputs_true: desired output
        outputs_pred: real output
        labels: names of classes
        outputs_outliers: output for outliers
    Returns:
        FPR, TPR, area under the ROC curve
    """
    n_classes = len(labels)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    y_score = threshold_output(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes)]
    y_score_classes = [[] for _ in range(n_classes)]
    for i, correctness, score in zip(y_ideal, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    
    if outputs_outliers is not None:
        y_real_outliers = [np.argmax(o) for o in outputs_outliers]
        y_score_outliers = rejection_score(outputs_outliers, 0)
        for prediction, score in zip(y_real_outliers, y_score_outliers):
            y_true_classes[prediction].append(False)
            y_score_classes[prediction].append(score)
    
    for i in range(n_classes):
        if len(y_true_classes) > 1:
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true_classes[i], y_score_classes[i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        else:
            fpr[i] = tpr[i] = roc_auc[i] = None
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(outputs_true.ravel(), outputs_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if fpr[i] is not None and tpr[i] is not None: 
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    #
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc

def calc_precision_recall(y, outputs):
    """
    Check if needed
    """
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    predictions = [np.argmax(o) for o in outputs]
    y_true = [a==b for a,b in zip(np.array(y), np.array(predictions))]
    
    for i in [0,1,2]:
        y_score = rejection_score(outputs, i)
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true, y_score)
        average_precision[i] = metrics.average_precision_score(y_true, y_score)
        
    return precision, recall, average_precision

if __name__ == '__main__':
    calc_roc_multiple(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
                      np.array([[0.9, 0.1], [0.4, 0.6], [0.5, 0.3], [0.1, 0.9]]),
                      np.array(['0', '1']))
    
    #y_true = np.array([True, True, False, False])
    #y_score = np.array([0.9, 0.6, 0.7, 0.1])
    
    #fpr1, tpr1, thr1 = metrics.roc_curve(y_true, y_score)
    #print(fpr1)
    #print(tpr1)
    #print(thr1)
    
    #fpr2, tpr2, thr2 = my_roc_curve(y_true, y_score)
    #print(fpr2)
    #print(tpr2)
    #print(thr2)
    