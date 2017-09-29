'''
Created on Jul 11, 2017

@author: anton
'''

from klnvch.rejection_option.thresholds import Thresholds as thr
import numpy as np


###############################################################################
#
# All helper functions to build is below
#
# y_true = true:  pattern is correctly classified
# y_true = false: pattern is misclassified or outlier
#
###############################################################################
# SINGLE THRESHOLD, NO REJECT OUTPUT
###############################################################################
def sc(outputs_true, outputs_pred, outputs_outl, threshold_func):
    """
    |    *    |    0    |    1    |    R    |
    |---------|---------|---------|---------|
    |    0    |    T    |    F    |    F    |
    |---------|---------|---------|---------|
    |    1    |    F    |    T    |    F    |
    """
    y_true = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a == b for a, b in zip(y_true, y_real)]
    
    if outputs_outl is not None:
        outputs_pred = np.concatenate([outputs_pred, outputs_outl])
        y_true = y_true + [False] * outputs_outl.shape[0]
    
    y_score = threshold_func(outputs_pred)
    
    return y_true, y_score
###############################################################################
# SINGLE THRESHOLD, WITH REJECT OUTPUT
###############################################################################
def sc_ir(outputs_true, outputs_pred):
    """
    |    *    |    0    |    1    |    R    |
    |---------|---------|---------|---------|
    |    0    |    T    |    F    |    F    |
    |---------|---------|---------|---------|
    |    1    |    F    |    T    |    F    |
    |---------|---------|---------|---------|
    |    R    |    x    |    x    |    x    |
    """
    y_true = outputs_true.argmax(axis=1)
    y_real = outputs_pred[:, :-1].argmax(axis=1)
    y_true = [a == b for a, b in zip(y_true, y_real)]
    
    return y_true, outputs_pred

def score_outp_ir(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr.thr_output_ignore_reject(outputs_pred)
    return y_true, y_score, 'Single output threshold'

def score_diff_ir(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr.thr_diff_ignore_reject(outputs_pred)
    return y_true, y_score, 'Single differential threshold'

def score_rati_ir(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr.thr_ratio_ignore_reject(outputs_pred)
    return y_true, y_score, 'Single ratio threshold'

def score_outp_or(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr.thr_only_reject(outputs_pred)
    return y_true, y_score, 'Rejection output threshold'

def score_diff_r(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr.thr_diff_reject(outputs_pred)
    return y_true, y_score, 'Rejection differential threshold'

def score_rati_r(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr.thr_ratio_reject(outputs_pred)
    return y_true, y_score, 'Rejection ratio threshold'
###############################################################################
# MULTIPLE THRESHOLDS, NO REJECT OUTPUT
###############################################################################
def sc_m(outputs_true, outputs_pred, outputs_outl, threshold_func):
    """
    Returns arrays of a[i,j] - i is a class and j is a sample
    """
    # find class indecies
    y_true = outputs_true.argmax(axis=1)
    y_pred = outputs_pred.argmax(axis=1)
    
    # prepare data for sklearn metrics
    y_true = [a == b for a, b in zip(y_true, y_pred)]
    y_score = threshold_func(outputs_pred)
    
    # split in classes
    n_classes = outputs_true.shape[1]
    y_m_true = [[] for _ in range(n_classes)]
    y_m_score = [[] for _ in range(n_classes)]
    
    for i, correctness, score in zip(y_pred, y_true, y_score):
        y_m_true[i].append(correctness)
        y_m_score[i].append(score)
    
    # add outliers
    if outputs_outl is not None:
        y_pred = outputs_outl.argmax(axis=1)
        y_score = threshold_func(outputs_outl)
        
        for i, score in zip(y_pred, y_score):
            y_m_true[i].append(False)
            y_m_score[i].append(score)
    
    return y_m_true, y_m_score
###############################################################################
# MULTIPLE THRESHOLDS, WITH REJECT OUTPUT
###############################################################################
def score_outp_ir_m(n_classes, outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred[:, :-1].argmax(axis=1)
    y_true = [a == b for a, b in zip(y_ideal, y_real)]
    y_score = thr.thr_output_ignore_reject(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes - 1)]
    y_score_classes = [[] for _ in range(n_classes - 1)]
    
    for i, correctness, score in zip(y_real, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    
    # if outputs_outl is not None:
    #    for o in outputs_outl:
    #        i = o.argmax()
    #        y_true_classes[i].append(False)
    #        y_score_classes[i].append(o.max())
    
    return y_true_classes, y_score_classes, \
            'Multiple rejection output threshold'

def score_diff_r_m(n_classes, outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred[:, :-1].argmax(axis=1)
    y_true = [a == b for a, b in zip(y_ideal, y_real)]
    y_score = thr.thr_diff_reject(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes - 1)]
    y_score_classes = [[] for _ in range(n_classes - 1)]
    
    for i, correctness, score in zip(y_real, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    
    # if outputs_outl is not None:
    #    for o in outputs_outl:
    #        i = o.argmax()
    #        y_true_classes[i].append(False)
    #        y_score_classes[i].append(o.max())
    
    return y_true_classes, y_score_classes, \
            'Multiple differential rejection threshold'

class ScoringFunc:
    ###################    Single threshold    ################################
    @staticmethod
    def score_outp(outputs_true, outputs_pred, outputs_outl):
        y_true, y_score = sc(outputs_true, outputs_pred, outputs_outl,
                             thr.thr_output)
        return y_true, y_score, 'Single output threshold'
    
    @staticmethod
    def score_diff(outputs_true, outputs_pred, outputs_outl):
        y_true, y_score = sc(outputs_true, outputs_pred, outputs_outl,
                             thr.thr_diff)
        return y_true, y_score, 'Single differential threshold'
    
    @staticmethod
    def score_rati(outputs_true, outputs_pred, outputs_outl):
        y_true, y_score = sc(outputs_true, outputs_pred, outputs_outl,
                             thr.thr_ratio)
        return y_true, y_score, 'Single ratio threshold'
    
    ###################    Multiple thresholds    #############################
    @staticmethod
    def score_outp_m(outputs_true, outputs_pred, outputs_outl):
        y_m_true, y_m_score = sc_m(outputs_true, outputs_pred, outputs_outl,
                                   thr.thr_output)
        return y_m_true, y_m_score, 'Multiple output threshold'
    
    @staticmethod
    def score_diff_m(outputs_true, outputs_pred, outputs_outl):
        y_m_true, y_m_score = sc_m(outputs_true, outputs_pred, outputs_outl,
                                   thr.thr_diff)
        return y_m_true, y_m_score, 'Multiple differential threshold'
    
    @staticmethod
    def score_rati_m(outputs_true, outputs_pred, outputs_outl):
        y_m_true, y_m_score = sc_m(outputs_true, outputs_pred, outputs_outl,
                                   thr.thr_ratio)
        return y_m_true, y_m_score, 'Multiple ratio threshold'
