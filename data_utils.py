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
    y_true = [a==b for a,b in zip(np.array(y), np.array(predictions))]
    
    for i in [0,1,2]:
        y_score = rejection_score(outputs, i)
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    return fpr, tpr, roc_auc

def calc_precision_recall(y, outputs):
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

def roc_curve(y_true, y_score, pos_label=1, drop_intermediate=True):
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr, thresholds

def _binary_clf_curve(y_true, y_score, pos_label=1):
    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]