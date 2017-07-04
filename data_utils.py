'''
Created on Oct 26, 2016

@author: anton
'''
import numpy as np
from sklearn import metrics
from scipy import interp
import itertools
import time
from thresholds import rejection_score, thr_output

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

def roc_s_thr(outputs_true, outputs_pred, outputs_outl, scores):
    """
    Calcs binary ROC curve or for single threshold
    Args:
        outputs_true : desired output
        outputs_pred : real output
        outputs_outliers : output for outliers
    Returns:
        FPR, TPR, AUC
    """
    def calc(outputs_true, outputs_pred, outputs_outl, score):
        y_true, y_score, label = score(outputs_true, outputs_pred, outputs_outl)
        fpr, tpr, thr = metrics.roc_curve(y_true, y_score)
        # add 0 and 1 to get full curve
        fpr = np.concatenate(([0.], fpr, [1.]))
        tpr = np.concatenate(([0.], tpr, [1.]))
        roc_auc = metrics.auc(fpr, tpr)
        return fpr, tpr, thr, roc_auc, label
    
    result = [calc(outputs_true, outputs_pred, outputs_outl, i) for i in scores]
    return np.array(result)

def roc_m_thr(n_classes, outputs_true, outputs_pred, outputs_outl, scores):
    """Calculates ROC for multiple output thresholds
    
    Output i with threshold T_i must deal with:
        - patterns from class i, that are classified as i
        - patterns from class j, that are classified as i
        - outliers,              that are classified as i
    
    Caclulation below are very expensive
    [871 x 729] size takes 127.252952 seconds
    [421 x 369] size takes  16.488903 seconds
    [366 x 424] size takes 13.825663 seconds
    """
    def clean_dots(xs, ys, thr, comparison):
        xs, ys, thr = zip(*sorted(zip(xs, ys, thr)))
        new_xs = [xs[0]]
        new_ys = [ys[0]]
        new_thr = [thr[0]]
        for x, y, t in zip(xs, ys, thr):
            if new_xs[-1] == x:
                new_ys[-1] = comparison(new_ys[-1], y)
                if new_ys[-1] == y: new_thr[-1] = t
            else:
                new_xs.append(x)
                new_ys.append(y)
                new_thr.append(t)
        xs = np.concatenate(([0.], new_xs, [1.]))
        ys = np.concatenate(([0.], new_ys, [1.]))
        stupid_thr = len(thr[0])*[None]
        thr = np.concatenate(([stupid_thr], new_thr, [stupid_thr]))
        return xs, ys, thr
    
    def calc(n_classes, outputs_true, outputs_pred, outputs_outl, score):
        start_time = time.time()
        
        y_true_classes, y_score_classes, label = \
            score(n_classes, outputs_true, outputs_pred, outputs_outl)
        
        lenghts = np.array([len(i) for i in y_true_classes])
        lenghts = np.array2string(lenghts, separator=' x ')
        print('roc_m_thr: {:s} size'.format(lenghts))
        
        fpr = []
        tpr = []
        thr = []
        
        for thresholds in itertools.product(*y_score_classes):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            
            for i, t in enumerate(thresholds):
                for corect, score in zip(y_true_classes[i], y_score_classes[i]):
                    if corect:
                        if score >= t: tp += 1
                        else:          fn += 1
                    else:
                        if score >= t: fp += 1
                        else:          tn += 1
            
            if fp + tn > 0 and tp + fn > 0: 
                fpr.append(fp / (fp + tn))
                tpr.append(tp / (tp + fn))
                thr.append(thresholds)
        
        tpr, fpr, thr = clean_dots(tpr, fpr, thr, min)
        fpr, tpr, thr = clean_dots(fpr, tpr, thr, max)
        
        print('roc_m_thr: {:9f} seconds'.format(time.time() - start_time))
        
        return fpr, tpr, thr, metrics.auc(fpr, tpr), label
    
    result = [calc(n_classes, outputs_true, outputs_pred, outputs_outl, i) 
              for i in scores]
    
    return np.array(result)

def calc_roc_multiclass(outputs_true, outputs_pred, n_classes, outputs_outl=None):
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
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    y_score = thr_output(outputs_pred)
    
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
        if len(y_true_classes[i]) > 1:
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true_classes[i],
                                                  y_score_classes[i],
                                                  True)
            # add 0 and 1 to get full curve
            fpr[i] = np.concatenate(([0.], fpr[i], [1.]))
            tpr[i] = np.concatenate(([0.], tpr[i], [1.]))
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        else:
            fpr[i] = tpr[i] = []
            roc_auc[i] = None
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(outputs_true.ravel(), outputs_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if roc_auc[i] is not None: 
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