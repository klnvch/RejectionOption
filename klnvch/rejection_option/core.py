'''
Created on Jun 26, 2017

@author: anton
'''
from numpy import nan
from sklearn import metrics

from klnvch.rejection_option.plots import plot_confusion_matrix
from klnvch.rejection_option.plots import plot_curves
from klnvch.rejection_option.plots import plot_decision_regions
from klnvch.rejection_option.plots import plot_multiclass_curve
from klnvch.rejection_option.scoring import ScoringFunc as score_func
from klnvch.rejection_option.thresholds import Thresholds as thr
from klnvch.rejection_option.utils import calc_multiclass_curve
from klnvch.rejection_option.utils import calc_s_thr, check_nan
from klnvch.rejection_option.utils import validate_classes
import numpy as np


class RejectionOption:
    
    def __init__(self, clf, n_classes, rc=False):
        """
        Args:
            clf:    classifier
            n_classes:    number of class
            rc:    rejection class
        """
        assert clf is not None
        assert rc is True or rc is False
        assert n_classes > 0
        
        self.clf = clf
        self.rc = rc
        self.n_classes = n_classes
    
    def init(self, labels, x, y, outliers=None):
        assert x is not None and y is not None
        assert x.shape[0] == y.shape[0]
        
        self.labels = labels
        self.x = x
        self.y = y
        self.outputs_true = y
        self.outputs_pred = self.clf.predict_proba(x)
        self.outputs_outl = self.clf.predict_proba(outliers)
        
        if self.rc:
            self.outputs_true = self.outputs_true[:, :-1]
            self.outputs_pred = self.outputs_pred[:, :-1]
            self.outputs_outl = self.outputs_outl[:, :-1]
            self.labels = labels[:-1]
            self.n_classes -= 1
    
    def calc_metrics(self):
        if check_nan(self.outputs_pred):
            print('outputs have nan')
            return nan, nan, nan, nan, nan, nan
        
        # SOT - Single 0utput Threshold
        y_true, y_score, label = score_func.score_outp(self.outputs_true,
                                                       self.outputs_pred,
                                                       self.outputs_outl)
        if validate_classes(y_true):
            auc_0 = metrics.roc_auc_score(y_true, y_score)
            if auc_0 < .5:  auc_0 = 1. - auc_0
        else:
            auc_0 = 1.0
        print('{:40s}: {:0.4f}'.format(label, auc_0))
        
        # SDT - Single Differential Threshold
        y_true, y_score, label = score_func.score_diff(self.outputs_true,
                                                       self.outputs_pred,
                                                       self.outputs_outl)
        if validate_classes(y_true):
            auc_1 = metrics.roc_auc_score(y_true, y_score)
            if auc_1 < .5:  auc_1 = 1. - auc_1
        else:
            auc_1 = 1.0
        print('{:40s}: {:0.4f}'.format(label, auc_1))
        
        # SRT - Single Ratio Threshold
        y_true, y_score, label = score_func.score_rati(self.outputs_true,
                                                       self.outputs_pred,
                                                       self.outputs_outl)
        if validate_classes(y_true):
            auc_2 = metrics.roc_auc_score(y_true, y_score)
            if auc_2 < .5:  auc_2 = 1. - auc_2
        else:
            auc_2 = 1.0
        print('{:40s}: {:0.4f}'.format(label, auc_2))
        
        # MOT - Multiple Output Thresholds
        y_m_true, y_m_score, label = score_func.score_outp_m(self.outputs_true,
                                                             self.outputs_pred,
                                                             self.outputs_outl)
        avg_auc_0 = 0
        for y_true, y_score in zip(y_m_true, y_m_score):
            if validate_classes(y_true):
                auc = metrics.roc_auc_score(y_true, y_score)
                if auc < .5:  auc = 1. - auc
                avg_auc_0 += auc
            else:
                avg_auc_0 += 1.0
        avg_auc_0 /= self.n_classes
        print('{:40s}: {:0.4f}'.format(label, avg_auc_0))
        
        # MOT - Multiple Differential Thresholds
        y_m_true, y_m_score, label = score_func.score_diff_m(self.outputs_true,
                                                             self.outputs_pred,
                                                             self.outputs_outl)
        avg_auc_1 = 0
        for y_true, y_score in zip(y_m_true, y_m_score):
            if validate_classes(y_true):
                auc = metrics.roc_auc_score(y_true, y_score)
                if auc < .5:  auc = 1. - auc
                avg_auc_1 += auc
            else:
                avg_auc_1 += 1.0
        avg_auc_1 /= self.n_classes
        print('{:40s}: {:0.4f}'.format(label, avg_auc_1))
        
        # MOT - Multiple Ratio Thresholds
        y_m_true, y_m_score, label = score_func.score_rati_m(self.outputs_true,
                                                             self.outputs_pred,
                                                             self.outputs_outl)
        avg_auc_2 = 0
        for y_true, y_score in zip(y_m_true, y_m_score):
            if validate_classes(y_true):
                auc = metrics.roc_auc_score(y_true, y_score)
                if auc < .5:  auc = 1. - auc
                avg_auc_2 += auc
            else:
                avg_auc_2 += 1.0
        avg_auc_2 /= self.n_classes
        print('{:40s}: {:0.4f}'.format(label, avg_auc_2))
        
        return auc_0, auc_1, auc_2, avg_auc_0, avg_auc_1, avg_auc_2
    
    def plot_confusion_matrix(self):
        plot_confusion_matrix(self.outputs_true, self.outputs_pred,
                              self.outputs_outl, self.labels,
                              error_threshold=None)
        
        if self.n_classes <= 10: return
        
        plot_confusion_matrix(self.outputs_true, self.outputs_pred,
                              self.outputs_outl, self.labels,
                              error_threshold=0.98)
        plot_confusion_matrix(self.outputs_true, self.outputs_pred,
                              self.outputs_outl, self.labels,
                              error_threshold=0.95)
        plot_confusion_matrix(self.outputs_true, self.outputs_pred,
                              self.outputs_outl, self.labels,
                              error_threshold=0.93)
        plot_confusion_matrix(self.outputs_true, self.outputs_pred,
                              self.outputs_outl, self.labels,
                              error_threshold=0.90)
    
    def plot_roc(self):
        scores_s = [score_func.score_outp,
                    score_func.score_diff,
                    score_func.score_rati]
        curves_s = calc_s_thr(self.outputs_true,
                             self.outputs_pred,
                             self.outputs_outl,
                             scores_s)
        plot_curves(curves_s)
    
    def plot_roc_precision_recall(self):
        scores_s = [score_func.score_outp,
                    score_func.score_diff,
                    score_func.score_rati]
        curves_s = calc_s_thr(self.outputs_true,
                             self.outputs_pred,
                             self.outputs_outl,
                             scores_s, curve_func='precision_recall')
        plot_curves(curves_s, curve_func='precision_recall')
    
    def plot_multiclass_roc(self):
        if self.n_classes <= 10:  recall_threshold = 1.0
        else:                     recall_threshold = 0.9
        
        # multiple output thresholds
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        score_func=score_func.score_outp_m,
                                        recall_threshold=recall_threshold,
                                        curve_func='roc')
        plot_multiclass_curve(x, y, v, self.labels)
        
        # multiple differential thresholds
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        score_func=score_func.score_diff_m,
                                        recall_threshold=recall_threshold,
                                        curve_func='roc')
        plot_multiclass_curve(x, y, v, self.labels)
        
        # multiple ratio thresholds
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        score_func=score_func.score_rati_m,
                                        recall_threshold=recall_threshold,
                                        curve_func='roc')
        plot_multiclass_curve(x, y, v, self.labels)
    
    def plot_multiclass_precision_recall(self):
        if self.n_classes <= 10:  recall_threshold = 1.0
        else:                     recall_threshold = 0.9
        
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        recall_threshold=recall_threshold,
                                        curve_func='precision_recall')
        plot_multiclass_curve(x, y, v, self.labels,
                              curve_func='precision_recall')
    
    def print_classification_report(self):
        y_true = self.outputs_true.argmax(axis=1)
        y_pred = self.outputs_pred.argmax(axis=1)
        report = metrics.classification_report(y_true, y_pred,
                                               None, self.labels)
        print(report)
    
    def plot_decision_regions(self):
        if self.x.shape[1] == 2:
            plot_decision_regions(self.x, self.y, self.clf, thr.thr_output,
                                  reject_output=self.rc)
            plot_decision_regions(self.x, self.y, self.clf, thr.thr_diff,
                                  reject_output=self.rc)
            plot_decision_regions(self.x, self.y, self.clf, thr.thr_ratio,
                                  reject_output=self.rc)
    
    def print_thresholds(self):
        if self.curves_m is None: return
        fpr = self.curves_m[0][0]
        tpr = self.curves_m[0][1]
        thr = self.curves_m[0][2]
        
        y_true = [a.argmax() == b.argmax() for a, b in zip(self.outputs, self.y)]
        if self.outputs_outl is not None:
            y_true = np.concatenate((y_true, [False] * self.outputs_outl.shape[0]))
        
        outputs = np.concatenate((self.outputs, self.outputs_outl))
        
        for t, x, y in zip(thr, fpr, tpr):
            if None in t: continue
            y_pred = [o.max() >= t[o.argmax()] for o in outputs]
            cm = metrics.confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            print(','.join([' %.4f' % a for a in t]) 
                  + ', {:d}, {:d}, {:d}, {:d}'.format(tp, fp, fn, tn))
        
        print('The end')
