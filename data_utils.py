'''
Created on Oct 26, 2016

@author: anton
'''
import numpy as np
from sklearn import metrics
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