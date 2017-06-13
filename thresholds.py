'''
Created on Jun 12, 2017

@author: anton
'''
import numpy as np
import heapq

def rejection_score(outputs, i):
    """
    Compute scores for single threshold
    Args:
        outputs: real outputs
        i: 0-output, 1-differential, 2-ratio thresholds
    Returns:
        scores according to outputs
    """
    if   i == 0: return thr_output(outputs)
    elif i == 1: return thr_diff(outputs)
    elif i == 2: return thr_ratio(outputs)
    else: raise ValueError('rejection method is wrong: ' + i)

def thr_output(outputs):
    return np.max(outputs, axis=1)

def thr_diff(outputs):
    return np.array([diff_two_max(o) for o in outputs])

def thr_ratio(outputs):
    return np.array([ratio_two_max(o) for o in outputs])

def thr_only_reject(outputs):
    return np.array([1.0 - o[-1] for o in outputs])

def thr_output_ignore_reject(outputs):
    return np.max(outputs[:,:-1], axis=1)

def thr_diff_ignore_reject(outputs):
    return np.array([diff_two_max(o) for o in outputs[:,:-1]])

def thr_ratio_ignore_reject(outputs):
    return np.array([ratio_two_max(o) for o in outputs[:,:-1]])

def thr_diff_reject(outputs):
    return np.array([(o[:-1].max() - o[-1] + 1.0) / 2.0 for o in outputs])

def thr_ratio_reject(outputs):
    return np.array([o[:-1].max() / min(o[-1], 0.0001) for o in outputs])

def diff_two_max(output):
    x = heapq.nlargest(2, output)
    return x[0] - x[1]

def ratio_two_max(output):
    x = heapq.nlargest(2, output)
    ratio = x[1] / x[0]
    if 0.0 <= ratio <= 1.0: return ratio
    else:                   return 0.0
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
def sc(outputs_true, outputs_pred, outputs_outl):
    """
    |    *    |    0    |    1    |    R    |
    |---------|---------|---------|---------|
    |    0    |    T    |    F    |    F    |
    |---------|---------|---------|---------|
    |    1    |    F    |    T    |    F    |
    """
    y_true = outputs_true.argmax(axis=1)
    y_real = outputs_pred.argmax(axis=1)
    y_true = [a==b for a,b in zip(y_true, y_real)]
    
    if outputs_outl is not None:
        outputs_pred = np.concatenate([outputs_pred, outputs_outl])
        y_true = y_true + [False] * outputs_outl.shape[0]
    
    return y_true, outputs_pred

def score_outp(outputs_true, outputs_pred, outputs_outl):
    y_true, outputs_pred = sc(outputs_true, outputs_pred, outputs_outl)
    y_score = thr_output(outputs_pred)
    return y_true, y_score, 'Single output threshold'

def score_diff(outputs_true, outputs_pred, outputs_outl):
    y_true, outputs_pred = sc(outputs_true, outputs_pred, outputs_outl)
    y_score = thr_diff(outputs_pred)
    return y_true, y_score, 'Single differential threshold'

def score_rati(outputs_true, outputs_pred, outputs_outl):
    y_true, outputs_pred = sc(outputs_true, outputs_pred, outputs_outl)
    y_score = thr_ratio(outputs_pred)
    return y_true, y_score, 'Single ratio threshold'
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
    y_real = outputs_pred[:,:-1].argmax(axis=1)
    y_true = [a==b for a,b in zip(y_true, y_real)]
    
    return y_true, outputs_pred

def score_outp_ir(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr_output_ignore_reject(outputs_pred)
    return y_true, y_score, 'Single output threshold'

def score_diff_ir(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr_diff_ignore_reject(outputs_pred)
    return y_true, y_score, 'Single differential threshold'

def score_rati_ir(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr_ratio_ignore_reject(outputs_pred)
    return y_true, y_score, 'Single ratio threshold'

def score_outp_or(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr_only_reject(outputs_pred)
    return y_true, y_score, 'Rejection output threshold'

def score_diff_r(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr_diff_reject(outputs_pred)
    return y_true, y_score, 'Rejection differential threshold'

def score_rati_r(outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    y_true, outputs_pred = sc_ir(outputs_true, outputs_pred)
    y_score = thr_ratio_reject(outputs_pred)
    return y_true, y_score, 'Rejection ratio threshold'
###############################################################################
# MULTIPLE THRESHOLDS, NO REJECT OUTPUT
###############################################################################
def score_outp_m(n_classes, outputs_true, outputs_pred, outputs_outl):
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
    
    return y_true_classes, y_score_classes, 'Multiple output threshold'
###############################################################################
# MULTIPLE THRESHOLDS, WITH REJECT OUTPUT
###############################################################################
def score_outp_ir_m(n_classes, outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred[:,:-1].argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    y_score = thr_output_ignore_reject(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes-1)]
    y_score_classes = [[] for _ in range(n_classes-1)]
    
    for i, correctness, score in zip(y_real, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    
    #if outputs_outl is not None:
    #    for o in outputs_outl:
    #        i = o.argmax()
    #        y_true_classes[i].append(False)
    #        y_score_classes[i].append(o.max())
    
    return y_true_classes, y_score_classes, \
            'Multiple rejection output threshold'

def score_diff_r_m(n_classes, outputs_true, outputs_pred, outputs_outl):
    assert outputs_outl is None
    
    y_ideal = outputs_true.argmax(axis=1)
    y_real = outputs_pred[:,:-1].argmax(axis=1)
    y_true = [a==b for a,b in zip(y_ideal, y_real)]
    y_score = thr_diff_reject(outputs_pred)
    
    y_true_classes = [[] for _ in range(n_classes-1)]
    y_score_classes = [[] for _ in range(n_classes-1)]
    
    for i, correctness, score in zip(y_real, y_true, y_score):
        y_true_classes[i].append(correctness)
        y_score_classes[i].append(score)
    
    #if outputs_outl is not None:
    #    for o in outputs_outl:
    #        i = o.argmax()
    #        y_true_classes[i].append(False)
    #        y_score_classes[i].append(o.max())
    
    return y_true_classes, y_score_classes, \
            'Multiple differential rejection threshold'