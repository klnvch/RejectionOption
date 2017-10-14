'''
Created on Jun 26, 2017

@author: anton
'''

from numpy import nan
from sklearn import metrics

from klnvch.rejection_option.plots import plot_confusion_matrix, plot_curves, \
    plot_multiclass_curve, plot_decision_regions
from klnvch.rejection_option.scoring import ScoringFunc as score_func
from klnvch.rejection_option.thresholds import Thresholds as thr
from klnvch.rejection_option.utils import calc_multiclass_curve, \
    calc_binary_auc_metric, calc_multiclass_auc_metric, check_nan, \
    calc_binary_curve, calc_accuracy
import numpy as np


class RejectionOption:
    
    def __init__(self, clf, labels, tst_x, tst_y, outliers=None, rc=False):
        assert tst_x is not None and tst_y is not None
        assert tst_x.shape[0] == tst_y.shape[0]
        assert clf is not None
        assert rc is True or rc is False
        
        self.labels = labels
        self.n_classes = len(labels)
        self.x = tst_x
        self.y = tst_y
        self.outputs_true = tst_y
        self.outputs_pred = clf.predict_proba(tst_x)
        self.outputs_outl = clf.predict_proba(outliers)
        
        self.rc = rc
        if self.rc:
            self.outputs_true = self.outputs_true[:, :-1]
            self.outputs_pred = self.outputs_pred[:, :-1]
            self.outputs_outl = self.outputs_outl[:, :-1]
            self.labels = labels[:-1]
            self.n_classes -= 1
    
    def set_verbosity(self, show, dir_path, suff_file):
        self.show = show
        if dir_path is None or suff_file is None:
            self.fig_path = None
            self.txt_path = None
        else:
            self.fig_path = dir_path + suff_file + '_{:s}.png'
            self.txt_path = dir_path + suff_file + '_report.txt'
    
    def calc_metrics(self):
        if check_nan(self.outputs_pred):
            print('outputs have nan')
            return nan, nan, nan, nan, nan, nan
        
        # Classifier accuracy
        tst_acc = calc_accuracy(self.outputs_true, self.outputs_pred)
        
        # SOT - Single 0utput Threshold
        auc_0, msg_0 = calc_binary_auc_metric(score_func.score_outp,
                                              self.outputs_true,
                                              self.outputs_pred,
                                              self.outputs_outl)
        # SDT - Single Differential Threshold
        auc_1, msg_1 = calc_binary_auc_metric(score_func.score_diff,
                                              self.outputs_true,
                                              self.outputs_pred,
                                              self.outputs_outl)
        # SRT - Single Ratio Threshold
        auc_2, msg_2 = calc_binary_auc_metric(score_func.score_rati,
                                              self.outputs_true,
                                              self.outputs_pred,
                                              self.outputs_outl)
        # MOT - Multiple Output Thresholds
        avg_auc_0, msg_3 = calc_multiclass_auc_metric(score_func.score_outp_m,
                                                      self.n_classes,
                                                      self.outputs_true,
                                                      self.outputs_pred,
                                                      self.outputs_outl)
        # MDT - Multiple Differential Thresholds
        avg_auc_1, msg_4 = calc_multiclass_auc_metric(score_func.score_diff_m,
                                                      self.n_classes,
                                                      self.outputs_true,
                                                      self.outputs_pred,
                                                      self.outputs_outl)
        # MRT - Multiple Ratio Thresholds
        avg_auc_2, msg_5 = calc_multiclass_auc_metric(score_func.score_rati_m,
                                                      self.n_classes,
                                                      self.outputs_true,
                                                      self.outputs_pred,
                                                      self.outputs_outl)
        
        acc_report = '{:40s}: {:0.4f}\n' \
                     '---------------------------------------------------' \
                     .format('Accuracy', tst_acc)
        report = msg_0 + msg_1 + msg_2 + msg_3 + msg_4 + msg_5
        print(acc_report)
        print(report)
        if self.txt_path is not None:
            with open(self.txt_path, 'a+') as f:
                print(acc_report, file=f)
                print(report, file=f)
        
        return tst_acc, auc_0, auc_1, auc_2, avg_auc_0, avg_auc_1, avg_auc_2
    
    def plot_confusion_matrix(self):
        if not self.show and self.fig_path is None: return
        if self.fig_path is None:   save_fig = None
        else:   save_fig = self.fig_path.format('confusion_matrix')
        
        plot_confusion_matrix(self.outputs_true, self.outputs_pred,
                              self.outputs_outl, self.labels,
                              error_threshold=None,
                              savefig=save_fig, show=self.show)
        
        if self.n_classes <= 10: return
        
        if self.show:
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
        if not self.show and self.fig_path is None: return
        if self.fig_path is None:   save_fig = None
        else:   save_fig = self.fig_path.format('single_roc')
        
        scores_s = [score_func.score_outp,
                    score_func.score_diff,
                    score_func.score_rati]
        curves_s = calc_binary_curve(self.outputs_true,
                                     self.outputs_pred,
                                     self.outputs_outl,
                                     scores_s)
        plot_curves(curves_s, savefig=save_fig, show=self.show)
    
    def plot_precision_recall(self):
        if not self.show and self.fig_path is None: return
        if self.fig_path is None:   save_fig = None
        else:   save_fig = self.fig_path.format('single_prr')
        
        scores_s = [score_func.score_outp,
                    score_func.score_diff,
                    score_func.score_rati]
        curves_s = calc_binary_curve(self.outputs_true,
                                     self.outputs_pred,
                                     self.outputs_outl,
                                     scores_s, curve_func='precision_recall')
        plot_curves(curves_s, curve_func='precision_recall',
                    savefig=save_fig, show=self.show)
    
    def plot_multiclass_roc(self):
        if not self.show and self.fig_path is None: return
        if self.fig_path is None:
            save_fig_0 = save_fig_1 = save_fig_2 = None
        else:
            save_fig_0 = self.fig_path.format('multi_roc_outp')
            save_fig_1 = self.fig_path.format('multi_roc_diff')
            save_fig_2 = self.fig_path.format('multi_roc_rati')
        
        if self.n_classes <= 10:  recall_threshold = 1.0
        else:                     recall_threshold = 0.9
        
        # multiple output thresholds
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        score_func=score_func.score_outp_m,
                                        recall_threshold=recall_threshold,
                                        curve_func='roc')
        plot_multiclass_curve(x, y, v, self.labels,
                              savefig=save_fig_0, show=self.show)
        
        # multiple differential thresholds
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        score_func=score_func.score_diff_m,
                                        recall_threshold=recall_threshold,
                                        curve_func='roc')
        plot_multiclass_curve(x, y, v, self.labels,
                              savefig=save_fig_1, show=self.show)
        
        # multiple ratio thresholds
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        score_func=score_func.score_rati_m,
                                        recall_threshold=recall_threshold,
                                        curve_func='roc')
        plot_multiclass_curve(x, y, v, self.labels,
                              savefig=save_fig_2, show=self.show)
    
    def plot_multiclass_precision_recall(self):
        if not self.show and self.fig_path is None: return
        if self.fig_path is None:   save_fig = None
        else:   save_fig = self.fig_path.format('multi_prr')
        
        if self.n_classes <= 10:  recall_threshold = 1.0
        else:                     recall_threshold = 0.9
        
        x, y, v = calc_multiclass_curve(self.outputs_true,
                                        self.outputs_pred,
                                        self.outputs_outl,
                                        recall_threshold=recall_threshold,
                                        curve_func='precision_recall')
        plot_multiclass_curve(x, y, v, self.labels,
                              curve_func='precision_recall',
                              savefig=save_fig, show=self.show)
    
    def print_classification_report(self):
        y_true = self.outputs_true.argmax(axis=1)
        y_pred = self.outputs_pred.argmax(axis=1)
        report = metrics.classification_report(y_true, y_pred,
                                               None, self.labels)
        
        print(report)
        if self.txt_path is not None:
            with open(self.txt_path, 'a+') as f:
                print(report, file=f)
    
    def plot_decision_regions(self, clf):
        if self.x.shape[1] == 2 and clf is not None:
            if not self.show and self.fig_path is None: return
            if self.fig_path is None:
                save_fig_0 = save_fig_1 = save_fig_2 = None
            else:
                save_fig_0 = self.fig_path.format('boundaries_outp')
                save_fig_1 = self.fig_path.format('boundaries_diff')
                save_fig_2 = self.fig_path.format('boundaries_rati')
            
            plot_decision_regions(self.x, self.y, clf, thr.thr_output,
                                  reject_output=self.rc,
                                  savefig=save_fig_0, show=self.show)
            plot_decision_regions(self.x, self.y, clf, thr.thr_diff,
                                  reject_output=self.rc,
                                  savefig=save_fig_1, show=self.show)
            plot_decision_regions(self.x, self.y, clf, thr.thr_ratio,
                                  reject_output=self.rc,
                                  savefig=save_fig_2, show=self.show)
    
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
