'''
Created on Jun 26, 2017

@author: anton
'''
from data_utils import roc_s_thr, roc_m_thr
import numpy as np
from graphics import plot_roc_curves
from sklearn import metrics
from klnvch.rejection_option.plots import plot_confusion_matrix
from klnvch.rejection_option.plots import plot_multiclass_curve
from klnvch.rejection_option.utils import calc_multiclass_curve
from klnvch.rejection_option.utils import validate_classes
from klnvch.rejection_option.scoring import ScoringFunc as score_func

def get_labels(n_classes, rc=False, thresholds='all'):
    """
    """
    assert n_classes is not None
    assert rc is True or rc is False
    assert thresholds in ['all','simple']
    
    if rc:
        if thresholds == 'all':
            return 'SOT,SDT,SRT,RT,SDRT,SRRT,MOT,MDRT'
        else:
            return 'SOT,SDT,SRT,RT,SDRT,SRRT'
    else:
        if thresholds == 'all':
            result = 'SOT,SDT,SRT,MOT,'
        else:
            result = 'SOT,SDT,SRT'
        for i in range(n_classes): result += 'OT C{:d},'.format(i)
        result += 'Micro,Macro'
        return result

class RejectionOption:
    
    def __init__(self, clf, n_classes, rc=False, thresholds='all'):
        """
        Args:
            thresholds:
                'all' - all thresholds
                'simple' - only single thresholds 
        """
        assert clf is not None
        assert rc is True or rc is False
        assert n_classes > 0
        assert thresholds in ['all','simple']
        
        self.clf = clf
        self.rc = rc
        self.n_classes = n_classes
        self.thresholds = thresholds
    
    def evaluate(self, x, y, outliers=None, output='csv'):
        """Computes all 
        """
        assert x is not None and y is not None
        assert x.shape[0] == y.shape[0]
        assert output == 'csv'
        
        if self.rc: return self.eval_rc(x, y)
        else:       return self.eval(x, y, outliers)
    
    def eval(self, x, y, outliers=None):
        outputs = self.clf.predict_proba(x)
        outputs_outl = self.clf.predict_proba(outliers)
        
        self.y = y
        self.outputs = outputs
        self.outputs_outl = outputs_outl
        
        scores_s = [score_func.score_outp, score_func.score_diff,
                    score_func.score_rati]
        self.curves_s = roc_s_thr(y, outputs, outputs_outl, scores_s)
        
        if self.thresholds == 'all':
            scores_m = [score_func.score_outp_m]
            self.curves_m = roc_m_thr(self.n_classes, y, outputs,
                                      outputs_outl, scores_m)
            curves = np.concatenate([self.curves_s, self.curves_m])
        else:
            self.curves_m = None
            curves = self.curves_s
        
        self.curve_mc = calc_multiclass_curve(y, outputs, self.n_classes,
                                              outputs_outl)
        fpr_mc, tpr_mc, auc_mc = self.curve_mc
        
        for i in range(self.n_classes):
            curves = np.vstack([curves, [fpr_mc[i], tpr_mc[i],
                                         None, auc_mc[i], '']])
        
        if 'micro' in auc_mc and 'macro' in auc_mc:
            curves = np.vstack([curves, [fpr_mc['micro'], tpr_mc['micro'],
                                         None, auc_mc['micro'], '']])
            curves = np.vstack([curves, [fpr_mc['macro'], tpr_mc['macro'],
                                         None, auc_mc['macro'], '']])
        
        aucs = curves[:,3]
        
        print(aucs)
        return ','.join([' %.5f' % num for num in aucs])
    
    def eval_rc(self, x, y):
        
        outputs = self.clf.predict_proba(x)
        
        scores_s = [score_outp_ir, score_diff_ir, score_rati_ir,
                  score_outp_or, score_diff_r, score_rati_r]
        self.curves_s = roc_s_thr(y, outputs, None, scores_s)
        
        if self.thresholds == 'all':
            scores_m = [score_outp_ir_m, score_diff_r_m]
            self.curves_m = roc_m_thr(self.n_classes, y, outputs,
                                      None, scores_m)
            curves = np.concatenate([curves_s, curves_m])
        else:
            self.curves_m = None
            curves = self.curves_s
        
        aucs = curves[:,3]
        
        return ','.join([' %.5f' % num for num in aucs])
    
    def calc_metrics(self):
        # SOT - Single 0utput Threshold
        y_true, y_score, label = \
                score_func.score_outp(self.y, self.outputs, self.outputs_outl)
        auc = metrics.roc_auc_score(y_true, y_score)
        print('{:40s}: {:0.4f}'.format(label, auc))
        
        # SDT - Single Differential Threshold
        y_true, y_score, label = \
                score_func.score_diff(self.y, self.outputs, self.outputs_outl)
        auc = metrics.roc_auc_score(y_true, y_score)
        print('{:40s}: {:0.4f}'.format(label, auc))
        
        # SRT - Single Ratio Threshold
        y_true, y_score, label = \
                score_func.score_rati(self.y, self.outputs, self.outputs_outl)
        auc = metrics.roc_auc_score(y_true, y_score)
        print('{:40s}: {:0.4f}'.format(label, auc))
        
        # MOT - Multiple Output Thresholds
        y_m_true, y_m_score, label = \
                score_func.score_outp_m(self.y, self.outputs, self.outputs_outl)
        avg_auc = 0
        for y_true, y_score in zip(y_m_true, y_m_score):
            if validate_classes(y_true):
                avg_auc += metrics.roc_auc_score(y_true, y_score)
            else:
                avg_auc += 1.0
        print('{:40s}: {:0.4f}'.format(label, avg_auc / self.n_classes))
        
        # MOT - Multiple Differential Thresholds
        y_m_true, y_m_score, label = \
                score_func.score_diff_m(self.y, self.outputs, self.outputs_outl)
        avg_auc = 0
        for y_true, y_score in zip(y_m_true, y_m_score):
            if validate_classes(y_true):
                avg_auc += metrics.roc_auc_score(y_true, y_score)
            else:
                avg_auc += 1.0
        print('{:40s}: {:0.4f}'.format(label, avg_auc / self.n_classes))
        
        # MOT - Multiple Ratio Thresholds
        y_m_true, y_m_score, label = \
                score_func.score_rati_m(self.y, self.outputs, self.outputs_outl)
        avg_auc = 0
        for y_true, y_score in zip(y_m_true, y_m_score):
            if validate_classes(y_true):
                avg_auc += metrics.roc_auc_score(y_true, y_score)
            else:
                avg_auc += 1.0
        print('{:40s}: {:0.4f}'.format(label, avg_auc / self.n_classes))
    
    def plot_confusion_matrix(self, labels):
        plot_confusion_matrix(self.y, self.outputs, self.outputs_outl, labels,
                              error_threshold=None)
        plot_confusion_matrix(self.y, self.outputs, self.outputs_outl, labels,
                              error_threshold=0.98)
        plot_confusion_matrix(self.y, self.outputs, self.outputs_outl, labels,
                              error_threshold=0.95)
        plot_confusion_matrix(self.y, self.outputs, self.outputs_outl, labels,
                              error_threshold=0.93)
        plot_confusion_matrix(self.y, self.outputs, self.outputs_outl, labels,
                              error_threshold=0.90)
    
    def plot(self):
        if self.curves_m is None:
            curves = self.curves_s
        else:
            curves = np.concatenate([self.curves_s, self.curves_m])
        plot_roc_curves(curves)
    
    def plot_multiclass_roc(self, labels):
        x, y, v = calc_multiclass_curve(self.y, self.outputs,
                                              self.n_classes,
                                              self.outputs_outl)
        plot_multiclass_curve(x, y, v, labels)
    
    def plot_multiclass_precision_recall(self, labels):
        x, y, v = calc_multiclass_curve(self.y, self.outputs,
                                        self.n_classes, self.outputs_outl,
                                        curve_func='precision_recall')
        plot_multiclass_curve(x, y, v, labels, curve_func='precision_recall')
    
    def print_classification_report(self, labels):
        y_true = self.y.argmax(axis=1)
        y_pred = self.outputs.argmax(axis=1)
        report = metrics.classification_report(y_true, y_pred, None, labels)
        print(report)
    
    def print_thresholds(self):
        if self.curves_m is None: return
        fpr = self.curves_m[0][0]
        tpr = self.curves_m[0][1]
        thr = self.curves_m[0][2]
        
        y_true = [a.argmax() == b.argmax() for a,b in zip(self.outputs, self.y)]
        if self.outputs_outl is not None:
            y_true = np.concatenate((y_true, [False]*self.outputs_outl.shape[0]))
        
        outputs = np.concatenate((self.outputs, self.outputs_outl))
        
        for t, x, y in zip(thr, fpr, tpr):
            if None in t: continue
            y_pred = [o.max() >= t[o.argmax()] for o in outputs]
            cm = metrics.confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            print(','.join([' %.4f' % a for a in t]) 
                  + ', {:d}, {:d}, {:d}, {:d}'.format(tp, fp, fn, tn))
        
        print('The end')

if __name__ == '__main__':
    y_true = np.random.randint(2, size=(100,4))
    y_score = np.random.rand(100,4)
    y_score[20,3] = None
    auc = metrics.roc_auc_score(y_true, y_score, None)
    print(auc)