'''
Created on Jul 10, 2017

@author: anton
'''

import numpy as np
from sklearn import metrics
from scipy import interp
from klnvch.rejection_option.scoring import ScoringFunc as score_func

def validate_classes(y_true):
    return len(np.unique(y_true)) == 2

def calc_multiclass_curve(outputs_true, outputs_pred, outputs_outl=None,
                          recall_threshold = 0.9,
                          score_func=score_func.score_outp_m, avg=False,
                          curve_func = 'roc'):
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
        return np.count_nonzero(a==True) / len(a)
    
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
    
    y_m_true, y_m_score, _ =score_func(outputs_true, outputs_pred, outputs_outl)
    
    n_classes = outputs_true.shape[1]
    for i in range(n_classes):
        # skip if few outputs or nothing to reject
        if (len(y_m_true[i]) > 1 and
            calc_recall(y_m_true[i]) <= recall_threshold):
            
            fpr[i], tpr[i], _ = curve_func(y_m_true[i],
                                           y_m_score[i],
                                           True)
            # add 0 and 1 to get full curve
            fpr[i] = np.concatenate(([0.], fpr[i], [1.]))
            tpr[i] = np.concatenate(([0.], tpr[i], [1.]))
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