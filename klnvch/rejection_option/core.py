'''
Created on Jun 26, 2017

@author: anton
'''
from thresholds import score_rati, score_diff, score_outp, score_outp_m,\
    score_diff_ir, score_outp_ir, score_rati_ir, score_diff_r, score_outp_or,\
    score_rati_r, score_diff_r_m, score_outp_ir_m
from data_utils import roc_s_thr, roc_m_thr, calc_roc_multiclass
import numpy as np

def get_labels(n_classes, rc=False, thresholds='all'):
    """
    """
    assert n_classes is not None
    assert rc is True or rc is False
    assert thresholds == 'all'
    
    if rc:
        return 'SOT,SDT,SRT,RT,SDRT,SRRT,MOT,MDRT'
    else:
        result = 'SOT,SDT,SRT,MOT,'
        for i in range(n_classes): result += 'OT C{:d},'.format(i)
        result += 'Micro,Macro'
        return result

class RejectionOption:
    
    def __init__(self, clf, n_classes, rc=False, thresholds='all'):
        """
        """
        assert clf is not None
        assert rc is True or rc is False
        assert n_classes > 0
        assert thresholds is not None
        
        self.clf = clf
        self.rc = rc
        self.n_classes = n_classes
    
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
        
        scores_s = [score_outp, score_diff, score_rati]
        curves_s = roc_s_thr(y, outputs, outputs_outl, scores_s)
        
        scores_m = [score_outp_m]
        curves_m = roc_m_thr(self.n_classes, y, outputs, outputs_outl, scores_m)
        
        curves = np.concatenate([curves_s, curves_m])
        
        fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(y, outputs,
                                                 self.n_classes, outputs_outl)
        
        for i in range(self.n_classes):
            curves = np.vstack([curves, [fpr_mc[i], tpr_mc[i], auc_mc[i], '']])
        
        curves = np.vstack([curves, [fpr_mc['micro'], tpr_mc['micro'], auc_mc['micro'], '']])
        curves = np.vstack([curves, [fpr_mc['macro'], tpr_mc['macro'], auc_mc['macro'], '']])
        
        aucs = curves[:,2]
        
        return ','.join([' %.5f' % num for num in aucs])
    
    def eval_rc(self, x, y):
        
        outputs = self.clf.predict_proba(x)
        
        scores_s = [score_outp_ir, score_diff_ir, score_rati_ir,
                  score_outp_or, score_diff_r, score_rati_r]
        curves_s = roc_s_thr(y, outputs, None, scores_s)
        
        scores_m = [score_outp_ir_m, score_diff_r_m]
        curves_m = roc_m_thr(self.n_classes, y, outputs, None, scores_m)
        
        curves = np.concatenate([curves_s, curves_m])
        
        aucs = curves[:,2]
        
        return ','.join([' %.5f' % num for num in aucs])