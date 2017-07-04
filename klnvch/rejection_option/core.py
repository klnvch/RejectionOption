'''
Created on Jun 26, 2017

@author: anton
'''
from thresholds import score_rati, score_diff, score_outp, score_outp_m,\
    score_diff_ir, score_outp_ir, score_rati_ir, score_diff_r, score_outp_or,\
    score_rati_r, score_diff_r_m, score_outp_ir_m
from data_utils import roc_s_thr, roc_m_thr, calc_roc_multiclass
import numpy as np
from graphics import plot_roc_curves, plot_multiclass_roc_curve,\
    plot_confusion_matrix
from sklearn.metrics.classification import confusion_matrix

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
        
        scores_s = [score_outp, score_diff, score_rati]
        self.curves_s = roc_s_thr(y, outputs, outputs_outl, scores_s)
        
        if self.thresholds == 'all':
            scores_m = [score_outp_m]
            self.curves_m = roc_m_thr(self.n_classes, y, outputs,
                                      outputs_outl, scores_m)
            curves = np.concatenate([self.curves_s, self.curves_m])
        else:
            self.curves_m = None
            curves = self.curves_s
        
        self.curve_mc = calc_roc_multiclass(y, outputs, self.n_classes,
                                            outputs_outl)
        fpr_mc, tpr_mc, auc_mc = self.curve_mc
        
        for i in range(self.n_classes):
            curves = np.vstack([curves, [fpr_mc[i], tpr_mc[i],
                                         None, auc_mc[i], '']])
        
        curves = np.vstack([curves, [fpr_mc['micro'], tpr_mc['micro'],
                                     None, auc_mc['micro'], '']])
        curves = np.vstack([curves, [fpr_mc['macro'], tpr_mc['macro'],
                                     None, auc_mc['macro'], '']])
        
        aucs = curves[:,3]
        
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
    
    def plot_confusion_matrix(self, labels):
        plot_confusion_matrix(self.outputs, self.y, labels, show=True)
    
    def plot(self):
        if self.curves_m is None:
            curves = self.curves_s
        else:
            curves = np.concatenate([self.curves_s, self.curves_m])
        plot_roc_curves(curves)
    
    def plot_multiclass(self, labels):
        fpr_mc, tpr_mc, auc_mc = self.curve_mc
        plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, labels)
    
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
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            print(','.join([' %.4f' % a for a in t]) 
                  + ', {:d}, {:d}, {:d}, {:d}'.format(tp, fp, fn, tn))
        
        print('The end')