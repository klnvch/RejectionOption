'''
Created on Jul 10, 2017

@author: anton
'''

import itertools
import time

from scipy import interp
from sklearn import metrics

from klnvch.rejection_option.scoring import ScoringFunc as score_func
import numpy as np


REPORT = 'ROC\n' \
            '{:s}' \
            '\t{:20s}: {:0.4f}\n' \
            '\t{:20s}: {:0.4f}\n' \
            '\t{:20s}: {:0.4f}\n' \
            '\t{:20s}: {:0.4f}\n' \
            'Precision-Recall\n' \
            '{:s}' \
            '\t{:20s}: {:0.4f}\n' \
            '\t{:20s}: {:0.4f}\n' \
            '\t{:20s}: {:0.4f}\n' \
            '\t{:20s}: {:0.4f}\n' \
            '{:40s}: {:0.4f}\n' \
            '---------------------------------------------------\n'


def validate_classes(y_true):
    return len(np.unique(y_true)) == 2
def check_nan(a):
    return np.isnan(a).any()

def calc_accuracy(outputs_true, outputs_pred):
    y_true = outputs_true.argmax(axis=1)
    y_pred = outputs_pred.argmax(axis=1)
    return metrics.accuracy_score(y_true, y_pred)

def calc_binary_auc_metric(score_func,
                           outputs_true, outputs_pred, outputs_outl):
    y_true, y_score, label = score_func(outputs_true,
                                        outputs_pred,
                                        outputs_outl)
    if validate_classes(y_true):
        auc = metrics.roc_auc_score(y_true, y_score)
        if auc < .5:
            auc = 1. - auc
    else:
        auc = 1.0
    # threshold examples
    thr_roc, y_pred = find_best_threshold(y_true, y_score, 'roc')
    q, w, e, r = calc_thresholds_metrics(y_true, y_pred,
                                         outputs_true, outputs_pred)
    
    thr_prr, y_pred = find_best_threshold(y_true, y_score, 'prr')
    a, s, d, f = calc_thresholds_metrics(y_true, y_pred,
                                         outputs_true, outputs_pred)
    
    report_thr_roc = '\t{:20s}: {:0.4f}\n'.format('Threshold', thr_roc)
    report_thr_prr = '\t{:20s}: {:0.4f}\n'.format('Threshold', thr_prr)
    
    report = REPORT.format(report_thr_roc,
                           'RO accuracy', q,
                           'RO F1-score', w,
                           'Rejecting Rate', e,
                           'ANN accuracy', r,
                           report_thr_prr,
                           'RO accuracy', a,
                           'RO F1-score', s,
                           'Rejecting Rate', d,
                           'ANN accuracy', f,
                           label, auc)
    
    return auc, report

def find_best_threshold(y_true, y_score, curve_func='roc'):
    """
    Find best thesholds and calc metrics for ANN and RO
    the best point is the closest to the top-left corner in the ROC space
    FPR = 0 and TPR = 1
    or
    Precision = 1 and Recall = 1
    """
    # some exceptions here
    y_true = np.array(y_true)
    if y_true.sum() == y_true.size:  # perfect ANN, accept everything
        return 0.0, y_true
    if y_true.sum() == 0:  # bad ANN, reject everything
        return 1.0, y_true
    
    # produce ROC curve
    if curve_func == 'roc':
        x, y, thr = metrics.roc_curve(y_true, y_score)
    elif curve_func == 'prr':
        x, y, thr = metrics.precision_recall_curve(y_true, y_score)
    
    # find best point
    if curve_func == 'roc':
        dists = [(_x ** 2 + _y ** 2) for _x, _y in zip(x, (1.0 - y))]
    elif curve_func == 'prr':
        dists = [(_x ** 2 + _y ** 2) for _x, _y in zip((1.0 - x), (1.0 - y))]
    idx = np.argmin(dists)
    best_threshold = thr[idx]
    # prepare RO decisions
    y_pred = y_score >= best_threshold
    
    return best_threshold, y_pred
    

def calc_thresholds_metrics(y_true, y_pred, outputs_true, outputs_pred):
    # RO accuracy
    ro_acc = metrics.accuracy_score(y_true, y_pred)
    
    # RO f1-score
    ro_f1_score = metrics.f1_score(y_true, y_pred)
    
    # Rejecting Rate
    rej_rate = 1.0 - y_pred.sum() / y_pred.size
    
    # ANN accuracy
    outputs_true_ = outputs_true[y_pred]
    outputs_pred_ = outputs_pred[y_pred]
    ann_acc = calc_accuracy(outputs_true_, outputs_pred_)
    
    return ro_acc, ro_f1_score, rej_rate, ann_acc
    
def calc_multiclass_auc_metric(score_func, n_classes,
                               outputs_true, outputs_pred, outputs_outl):
    y_m_true, y_m_score, label = score_func(outputs_true,
                                            outputs_pred,
                                            outputs_outl)
    
    thresholds_roc = np.empty((0), dtype=float)
    y_true_roc = np.empty((0), dtype=bool)
    y_pred_roc = np.empty((0), dtype=bool)
    
    thresholds_prr = np.empty((0), dtype=float)
    y_true_prr = np.empty((0), dtype=bool)
    y_pred_prr = np.empty((0), dtype=bool)
    
    avg_auc = 0
    for y_true, y_score in zip(y_m_true, y_m_score):
        if validate_classes(y_true):
            auc = metrics.roc_auc_score(y_true, y_score)
            if auc < .5:
                auc = 1. - auc
            avg_auc += auc
        else:
            avg_auc += 1.0
        # threshold examples
        # roc
        thr, y_pred = find_best_threshold(y_true, y_score)
        thresholds_roc = np.concatenate((thresholds_roc, [thr]))
        y_true_roc = np.concatenate((y_true_roc, y_true))
        y_pred_roc = np.concatenate((y_pred_roc, y_pred))
        
        # prr
        thr, y_pred = find_best_threshold(y_true, y_score, curve_func='prr')
        thresholds_prr = np.concatenate((thresholds_prr, [thr]))
        y_true_prr = np.concatenate((y_true_prr, y_true))
        y_pred_prr = np.concatenate((y_pred_prr, y_pred))
        
    
    
    # ROC
    report_thr_roc = ''
    for i, thr in enumerate(thresholds_roc):
        thr_label = 'Threshold {:d}'.format(i)
        report_thr_roc += '\t{:20s}: {:0.4f}\n'.format(thr_label, thr)
    q, w, e, r = calc_thresholds_metrics(y_true_roc, y_pred_roc,
                                         outputs_true, outputs_pred)
    # Precision-Recall
    report_thr_prr = ''
    for i, thr in enumerate(thresholds_prr):
        thr_label = 'Threshold {:d}'.format(i)
        report_thr_prr += '\t{:20s}: {:0.4f}\n'.format(thr_label, thr)
    a, s, d, f = calc_thresholds_metrics(y_true_prr, y_pred_prr,
                                         outputs_true, outputs_pred)
    
    avg_auc /= n_classes
    
    report = REPORT.format(report_thr_roc,
                           'RO accuracy', q,
                           'RO F1-score', w,
                           'Rejecting Rate', e,
                           'ANN accuracy', r,
                           report_thr_prr,
                           'RO accuracy', a,
                           'RO F1-score', s,
                           'Rejecting Rate', d,
                           'ANN accuracy', f,
                           label, avg_auc)
    
    return avg_auc, report

def calc_multiclass_curve(outputs_true, outputs_pred, outputs_outl=None,
                          recall_threshold=0.9,
                          score_func=score_func.score_outp_m, avg=False,
                          curve_func='roc'):
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
        curve: 'roc' or 'precision_recall'
    Returns:
        dictionary of FPR, TPR, AUC per class
        or
        dictionary of Precision, Recall, AUC per class
    """
    def calc_recall(a):
        a = np.array(a)
        return np.count_nonzero(a == True) / len(a)
    
    if curve_func == 'precision_recall':
        curve_func = metrics.precision_recall_curve
        metric_func = metrics.average_precision_score
    elif curve_func == 'roc':
        curve_func = metrics.roc_curve
        metric_func = metrics.roc_auc_score
    else: raise ValueError('wrong curve function')
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_m_true, y_m_score, _ = score_func(outputs_true, outputs_pred, outputs_outl)
    
    n_classes = outputs_true.shape[1]
    for i in range(n_classes):
        # skip if few outputs or nothing to reject
        if (len(y_m_true[i]) > 1 and validate_classes(y_m_true[i]) and
            calc_recall(y_m_true[i]) <= recall_threshold):
            
            fpr[i], tpr[i], _ = curve_func(y_m_true[i],
                                           y_m_score[i],
                                           True)
            # add 0 and 1 to get full curve
            # fpr[i] = np.concatenate(([0.], fpr[i], [1.]))
            # tpr[i] = np.concatenate(([0.], tpr[i], [1.]))
            roc_auc[i] = metric_func(y_m_true[i], y_m_score[i])
        else:
            fpr[i] = tpr[i] = []
            roc_auc[i] = 0
    
    if avg:
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = curve_func(outputs_true.ravel(),
                                                   outputs_pred.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            if roc_auc[i] > 0: 
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        #
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc

def calc_binary_curve(outputs_true, outputs_pred, outputs_outl, scores,
                      curve_func='roc'):
    """
    Calcs binary ROC curve or for single threshold
    Args:
        outputs_true : desired output
        outputs_pred : real output
        outputs_outliers : output for outliers
    Returns:
        FPR, TPR, AUC
    """
    def calc(outputs_true, outputs_pred, outputs_outl,
             score, curve_func='roc'):
        
        if curve_func == 'precision_recall':
            curve_func = metrics.precision_recall_curve
            metric_func = metrics.average_precision_score
        elif curve_func == 'roc':
            curve_func = metrics.roc_curve
            metric_func = metrics.roc_auc_score
        else: raise ValueError('wrong curve function')
        
        y_true, y_score, label = score(outputs_true, outputs_pred, outputs_outl)
        fpr, tpr, thr = curve_func(y_true, y_score)
        # add 0 and 1 to get full curve
        # fpr = np.concatenate(([0.], fpr, [1.]))
        # tpr = np.concatenate(([0.], tpr, [1.]))
        roc_auc = metric_func(y_true, y_score)
        return fpr, tpr, thr, roc_auc, label
    
    result = [calc(outputs_true, outputs_pred, outputs_outl, i, curve_func) 
              for i in scores]
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
        stupid_thr = len(thr[0]) * [None]
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
