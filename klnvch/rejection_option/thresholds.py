'''
Created on Jun 12, 2017

@author: anton
'''
import heapq

import numpy as np


def rejection_score(outputs, i):
    """
    Compute scores for single threshold
    Args:
        outputs: real outputs
        i: 0-output, 1-differential, 2-ratio thresholds
    Returns:
        scores according to outputs
    """
    if   i == 0: return Thresholds.thr_output(outputs)
    elif i == 1: return Thresholds.thr_diff(outputs)
    elif i == 2: return Thresholds.thr_ratio(outputs)
    else: raise ValueError('rejection method is wrong: ' + i)

def diff_two_max(output):
    x = heapq.nlargest(2, output)
    return x[0] - x[1]

def ratio_two_max(output):
    x = heapq.nlargest(2, output)
    ratio = 1.0 - x[1] / x[0]
    if 0.0 <= ratio <= 1.0: return ratio
    else:                   return 0.0

class Thresholds:
    @staticmethod
    def thr_output(outputs):
        return np.max(outputs, axis=1)
    
    @staticmethod
    def thr_diff(outputs):
        return np.array([diff_two_max(o) for o in outputs])
    
    @staticmethod
    def thr_ratio(outputs):
        return np.array([ratio_two_max(o) for o in outputs])
    
    @staticmethod
    def thr_only_reject(outputs):
        return np.array([1.0 - o[-1] for o in outputs])
    
    @staticmethod
    def thr_output_ignore_reject(outputs):
        return np.max(outputs[:, :-1], axis=1)
    
    @staticmethod
    def thr_diff_ignore_reject(outputs):
        return np.array([diff_two_max(o) for o in outputs[:, :-1]])
    
    @staticmethod
    def thr_ratio_ignore_reject(outputs):
        return np.array([ratio_two_max(o) for o in outputs[:, :-1]])
    
    @staticmethod
    def thr_diff_reject(outputs):
        return np.array([(o[:-1].max() - o[-1] + 1.0) / 2.0 for o in outputs])
    
    @staticmethod
    def thr_ratio_reject(outputs):
        return np.array([o[:-1].max() / min(o[-1], 0.0001) for o in outputs])
