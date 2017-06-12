'''
Created on Oct 26, 2016

@author: anton
'''
import numpy as np
import heapq
from sklearn import metrics
from scipy import interp
import itertools
import time

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
    return np.array([diff_two_max(o) for o in outputs])

def threshold_ratio(outputs):
    return np.array([ratio_two_max(o) for o in outputs])

def diff_two_max(output):
    x = heapq.nlargest(2, output)
    return x[0] - x[1]

def ratio_two_max(output):
    x = heapq.nlargest(2, output)
    ratio = 1.0 - x[1] / x[0]
    if 0.0 <= ratio <= 1.0: return ratio
    else:                   return 1.0

def calc_roc_binary(outputs_true, outputs_pred, outputs_outl=None):
    """
    Calcs binary ROC curve or for single threshold
    Args:
        outputs_true : desired output
        outputs_pred : real output
        outputs_outliers : output for outliers
    Returns:
        FPR, TPR, AUC
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    
    for i in [0, 1, 2]: # across rejection methods
        if outputs_outl is not None:
            outputs_pred = np.concatenate([outputs_pred, outputs_outl])
            y_true = y_true + [False] * outputs_outl.shape[0]
        
        y_score = rejection_score(outputs_pred, i)
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score)
        # add 0 and 1 to get full curve
        fpr[i] = np.concatenate(([0.], fpr[i], [1.]))
        tpr[i] = np.concatenate(([0.], tpr[i], [1.]))
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
        if fp + tn > 0: fpr.append(fp / (fp + tn))
        else:           fpr.append(1.0)
        if tp + fn > 0: tpr.append(tp / (tp + fn))
        else:           tpr.append(1.0)
    
    fpr, tpr, y_score = zip(*sorted(zip(fpr, tpr, y_score)))
    # add 0 and 1 to get full curve
    fpr = np.concatenate(([0.], fpr, [1.]))
    tpr = np.concatenate(([0.], tpr, [1.]))
    
    return fpr, tpr, y_score

def calc_roc_multiple(outputs_true, outputs_pred, labels, outputs_outl=None):
    """Calculates ROC for multiple output thresholds
    
    Output i with threshold T_i must deal with:
        - patterns from class i, that are classified as i
        - patterns from class j, that are classified as i
        - outliers,              that are classified as i
    
    Caclulation below are very expensive
    871 x 729 size takes 127.252952 seconds
    421 x 369 size takes  16.488903 seconds
    """
    start_time = time.time()
    n_classes = len(labels)
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    y_score = threshold_output(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes)]
    y_score_classes = [[] for _ in range(n_classes)]
    
    for i, correctness, score in zip(y_real, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    if outputs_outl is not None:
        for o in outputs_outl:
            i = o.argmax()
            y_true_classes[i].append(False)
            y_score_classes[i].append(o.max())
    
    print('calc_roc_multiple: {:d} x {:d} size'
          .format(len(y_true_classes[0]), len(y_true_classes[1])))
    
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
        
        if fp + tn > 0: fpr.append(fp / (fp + tn))
        else:           fpr.append(1.0)
        if tp + fn > 0: tpr.append(tp / (tp + fn))
        else:           tpr.append(1.0)
    
    tpr, fpr = clean_dots(tpr, fpr, min)
    fpr, tpr = clean_dots(fpr, tpr, max)
    #
    print('calc_roc_multiple: {:9f} seconds'.format(time.time() - start_time))
    
    return fpr, tpr, metrics.auc(fpr, tpr)

def clean_dots(xs, ys, comparison):
    xs, ys = zip(*sorted(zip(xs, ys)))
    new_xs = [xs[0]]
    new_ys = [ys[0]]
    for x, y in zip(xs, ys):
        if new_xs[-1] == x:
            new_ys[-1] = comparison(new_ys[-1], y) 
        else:
            new_xs.append(x)
            new_ys.append(y)
    xs = np.concatenate(([0.], new_xs, [1.]))
    ys = np.concatenate(([0.], new_ys, [1.]))
    return xs, ys

def calc_roc_multiclass(outputs_true, outputs_pred, labels, outputs_outl=None):
    """ Calcs binary ROC curve or for multiple output thresholds
    
    Output i with threshold T_i must deal with:
        - patterns from class i, that are classified as i
        - patterns from class j, that are classified as i
        - outliers,              that are classified as i
    
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
    for i, correctness, score in zip(y_real, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    if outputs_outl is not None:
        for o in outputs_outl:
            i = o.argmax()
            y_true_classes[i].append(False)
            y_score_classes[i].append(o.max())
    
    for i in range(n_classes):
        if len(y_true_classes) > 1:
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true_classes[i], y_score_classes[i])
            # add 0 and 1 to get full curve
            fpr[i] = np.concatenate(([0.], fpr[i], [1.]))
            tpr[i] = np.concatenate(([0.], tpr[i], [1.]))
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
    