'''
Created on May 31, 2017

@author: anton

Tests to generate results for Chapter 4 and Section 1 about generated classes
'''
from DataSet import DataSet
from MLP import MLP
from RBF import RBF
from graphics import plot_decision_regions, plot_confusion_matrix, \
    plot_2d_dataset, plot_multiclass_roc_curve,\
    plot_roc_curves
import time
import pandas as pd
import scipy.stats
import numpy as np
from thresholds import thr_output_ignore_reject,\
    thr_diff_ignore_reject, thr_ratio_ignore_reject,\
    thr_only_reject, thr_diff_reject, thr_ratio_reject,\
    thr_output, thr_diff, thr_ratio, score_outp_ir, score_diff_ir, score_rati_ir,\
    score_outp_or, score_diff_r, score_rati_r, score_outp_ir_m, score_outp,\
    score_diff, score_rati, score_outp_m, score_diff_r_m
from data_utils import roc_s_thr, calc_roc_multiclass,\
    roc_m_thr

FIG_HALF_SIZE = 4.1

def test_mlp_rc():
    ds = DataSet(12, 800, [0.25, 0.25, 0.5], add_noise=2)
    mlp = MLP(0.01, [ds.n_features, 64, ds.n_classes],
              ['sigmoid', 'sigmoid'], 'Adam', 0.0001, 40)
    result = mlp.train(20000, ds.trn, ds.vld, 1.0, 0, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = mlp.predict_proba(ds.tst.x)
    outputs_outl = mlp.predict_proba(ds.outliers)
    
    plot_2d_dataset(ds.tst.x, ds.tst.y)
    plot_confusion_matrix(outputs, ds.tst.y, ds.target_names, show=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_output_ignore_reject, show=True, reject_output=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_diff_ignore_reject, show=True, reject_output=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_ratio_ignore_reject, show=True, reject_output=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_only_reject, show=True, reject_output=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_diff_reject, show=True, reject_output=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_ratio_reject, show=True, reject_output=True)
    
    scores_s = [score_outp_ir, score_diff_ir, score_rati_ir,
              score_outp_or, score_diff_r, score_rati_r]
    curves_s = roc_s_thr(ds.tst.y, outputs, outputs_outl, scores_s)
    
    scores_m = [score_outp_ir_m, score_diff_r_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, outputs_outl, scores_m)
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]))

def test_mlp():
    #ds = DataSet(5, 1000, [0.2, 0.8])
    #ds = DataSet(5, 1000, [0.2, 0.2, 0.6])
    ds = DataSet(12, 800, [0.25, 0.25, 0.5], add_noise=1)
    #ds = DataSet(12, 600, [0.34, 0.66], add_noise=1)
    #ds = DataSet(5, 600, [0.25, 0.25, 0.6], add_noise=1)
    mlp = MLP(0.01, [ds.n_features, 16, ds.n_classes],
              ['relu', 'sigmoid'], 'Adam', 0.0, 40)
    result = mlp.train(20000, ds.trn, ds.vld, 1.0, 0, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = mlp.predict_proba(ds.tst.x)
    outputs_outl = mlp.predict_proba(ds.outliers)
    
    #plot_2d_dataset(ds.tst.x, ds.tst.y)
    #plot_confusion_matrix(outputs, ds.tst.y, ds.target_names)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_output)
    #plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_diff)
    #plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_ratio)
    
    scores_s = [score_outp, score_diff, score_rati]
    curves_s = roc_s_thr(ds.tst.y, outputs, outputs_outl, scores_s)
    
    scores_m = [score_outp_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, outputs_outl, scores_m)
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]))
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names, outputs_outl)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names)

def test_rfb():
    ds = DataSet(5, 1000, [0.2, 0.8])
    rbf = RBF(ds.n_features, 16, ds.n_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    
    score = rbf.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = rbf.predict_proba(ds.tst.x)
    outputs_outl = rbf.predict_proba(ds.outliers)
    
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, thr_output, step_size=0.05)
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, thr_diff, step_size=0.05)
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, thr_ratio, step_size=0.05)
    
    scores_s = [score_outp, score_diff, score_rati]
    curves_s = roc_s_thr(ds.tst.y, outputs, outputs_outl, scores_s)
    
    scores_m = [score_outp_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, outputs_outl, scores_m)
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]))
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names, outputs_outl)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names)

def test_unit(ds, ds_name, attempt, clf_name,
              n_hidden, beta, dropout, early_stopping, show=True):
    if clf_name == 'mlp-sigmoid':
        mlp = MLP(0.01, [ds.n_features, n_hidden, ds.n_classes],
                  ['sigmoid', 'sigmoid'], 'Adam', beta, 40)
    elif clf_name == 'mlp-softmax':
        mlp = MLP(0.01, [ds.n_features, n_hidden, ds.n_classes],
                  ['sigmoid', 'softmax'], 'Adam', beta, 40)
    result = mlp.train(20000, ds.trn, ds.vld, dropout, early_stopping, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = mlp.predict_proba(ds.tst.x)
    outputs_outl = mlp.predict_proba(ds.outliers)
    
    #file name example: boundaries_0_moons_000_mlp-sigmoid_006 
    savefig = 'tests/{0:s}/{2:s}/{3:s}_{0:s}_{1:03d}_{2:s}_{4:03d}' \
                .format(ds_name, attempt, clf_name, '{:s}', n_hidden)
    plot_confusion_matrix(outputs, ds.tst.y, ds.target_names,
                          savefig.format('confusion_matrix'), show)
    
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          savefig.format('boundaries_0'), show)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_diff,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          savefig.format('boundaries_1'), show)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          savefig.format('boundaries_2'), show)
    
    scores_s = [score_outp, score_diff, score_rati]
    curves_s = roc_s_thr(ds.tst.y, outputs, outputs_outl, scores_s)
    
    scores_m = [score_outp_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, outputs_outl, scores_m)
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]),
                    savefig.format('roc_m'), show)
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names,
                                                 outputs_outl)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names,
                              savefig.format('roc_m'), show)
    
    return '{:>8s}, {:4d}, {:>16}, {:4d}, ' \
        '{:9f}, {:9f}, {:4d}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f},' \
        '{:9f}, {:9f}, {:9f}, {:9f}' \
        ''.format(ds_name, attempt, clf_name, n_hidden,
                  beta, dropout, early_stopping,
                  result[2], result[3], result[4], score,
                  curves_s[0][2], curves_s[1][2], curves_s[2][2], curves_m[0][2],
                  auc_mc[0], auc_mc[1], auc_mc['micro'], auc_mc['macro'])

def test_block_RBF(ds_name, ds_id, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    colums = 'DS,Attempt,Units,Tst acc,' \
                    'SOT,SDT,SRT,MOT,OT C0,OT C1,Micro,Macro'
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, 1000, [0.2, 0.8])
        #ds = DataSet(ds_id, 600, [0.3333, 0.6667], add_noise=3)
        for param in params:
            n_hidden = param
            msg = test_unit_RBF(ds, ds_name, attempt, n_hidden)
            with open(filename, 'a+') as f:
                print(msg, file=f)

def test_unit_RBF(ds, ds_name, attempt, n_hidden):
    rbf = RBF(ds.n_features, n_hidden, ds.n_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    
    score = rbf.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = rbf.predict_proba(ds.tst.x)
    outputs_outl = rbf.predict_proba(ds.outliers)
    
    scores_s = [score_outp, score_diff, score_rati]
    curves_s = roc_s_thr(ds.tst.y, outputs, outputs_outl, scores_s)
    
    scores_m = [score_outp_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, outputs_outl, scores_m)
    
    _, _, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names, outputs_outl)
    
    return '{:>8s}, {:4d}, {:4d}, {:9f}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f},' \
        '{:9f}, {:9f}, {:9f}, {:9f}' \
        ''.format(ds_name, attempt, n_hidden, score,
                  curves_s[0][2], curves_s[1][2], curves_s[2][2], curves_m[0][2],
                  auc_mc[0], auc_mc[1], auc_mc['micro'], auc_mc['macro'])

def test_block(ds_name, ds_id, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    colums = 'DS,Attempt,Clf,Units,Beta,DO,ES,Loss,Trn acc,Vld acc,Tst acc,' \
                    'SOT,SDT,SRT,MOT,OT C0,OT C1,Micro,Macro'
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        #ds = DataSet(ds_id, 1200, [0.1666, 0.1666, 0.6668])
        ds = DataSet(ds_id, 800, [0.25, 0.25, 0.5], add_noise=1)
        for param in params:
            clf_name, n_hidden, beta, dropout, early_stopping = param
            msg = test_unit(ds, ds_name, attempt, clf_name, n_hidden, beta,
                            dropout, early_stopping, False)
            with open(filename, 'a+') as f:
                print(msg, file=f)

def test_unit_rc(ds, ds_name, attempt, clf_name,
              n_hidden, beta, dropout, early_stopping, show=True):
    if clf_name == 'mlp-sigmoid':
        mlp = MLP(0.01, [ds.n_features, n_hidden, ds.n_classes],
                  ['sigmoid', 'sigmoid'], 'Adam', beta, 40)
    elif clf_name == 'mlp-softmax':
        mlp = MLP(0.01, [ds.n_features, n_hidden, ds.n_classes],
                  ['sigmoid', 'softmax'], 'Adam', beta, 40)
    result = mlp.train(20000, ds.trn, ds.vld, dropout, early_stopping, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = mlp.predict_proba(ds.tst.x)
    
    scores_s = [score_outp_ir, score_diff_ir, score_rati_ir,
                score_outp_or, score_diff_r, score_rati_r]
    curves_s = roc_s_thr(ds.tst.y, outputs, None, scores_s)
    
    scores_m = [score_outp_ir_m, score_diff_r_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, None, scores_m)
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]), None, False)
    
    return '{:>8s}, {:4d}, {:>16}, {:4d}, ' \
        '{:9f}, {:9f}, {:4d}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f},' \
        '{:9f}, {:9f}, {:9f}, {:9f}' \
        ''.format(ds_name, attempt, clf_name, n_hidden,
                  beta, dropout, early_stopping,
                  result[2], result[3], result[4], score,
                  curves_s[0][2], curves_s[1][2], curves_s[2][2], curves_s[3][2],
                  curves_s[4][2], curves_s[5][2], curves_m[0][2], curves_m[1][2])

def test_block_rc(ds_name, ds_id, attempts, params):
    filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
    colums = 'DS,Attempt,Clf,Units,Beta,DO,ES,Loss,Trn acc,Vld acc,Tst acc,' \
                    'SOT,SDT,SRT,RT,SDRT,SRRT,MOT,MDRT'
    with open(filename, 'a+') as f:
        print(colums, file=f)
    
    for attempt in attempts:
        ds = DataSet(ds_id, 800, [0.25, 0.25, 0.5], add_noise=2)
        for param in params:
            clf_name, n_hidden, beta, dropout, early_stopping = param
            msg = test_unit_rc(ds, ds_name, attempt, clf_name, n_hidden, beta,
                               dropout, early_stopping, False)
            with open(filename, 'a+') as f:
                print(msg, file=f)

def wilcoxon_test():
    df = pd.read_csv('tests/moons/run_1496670425.csv')
    print(df.keys())
    
    mot = [df.loc[((df['Hidden units'] == i) & (df['Classifier'].str.strip() == 'mlp-softmax')), 'Multiple output threshold'].values for i in range(1,13)]
    
    A = [[scipy.stats.wilcoxon(mot[i], mot[j]).pvalue for i in range(12)] for j in range(12)]
    
    print('\n'.join([''.join(['{:4f} '.format(item) for item in row]) for row in A]))

if __name__ == '__main__':
    #test_rfb()
    #test_mlp()
    test_mlp_rc()
    """
    test_block('moons_overlapping', 5, range(1,2),
               [['mlp-sigmoid',  2, 0.0,  1.0, 0],
                ['mlp-sigmoid',  3, 0.0,  1.0, 0],
                ['mlp-sigmoid',  5, 0.0,  1.0, 0],
                ['mlp-sigmoid',  8, 0.0,  1.0, 0],
                ['mlp-sigmoid', 12, 0.0,  1.0, 0],
                ['mlp-sigmoid', 12, 0.0,  1.0, 100],
                ['mlp-sigmoid', 12, 1e-3, 1.0, 0],
                ['mlp-sigmoid', 12, 1e-4, 1.0, 0],
                ['mlp-sigmoid', 12, 1e-5, 1.0, 0],
                ['mlp-sigmoid', 12, 1e-6, 1.0, 0],
                ['mlp-sigmoid', 12, 1e-4, 1.0, 100],
                ['mlp-sigmoid', 12, 1e-4, 0.9, 0],
                ['mlp-sigmoid', 12, 1e-4, 0.8, 0],
                ['mlp-sigmoid', 12, 1e-4, 0.9, 100],
                ['mlp-softmax',  2, 0.0,  1.0, 0],
                ['mlp-softmax',  3, 0.0,  1.0, 0],
                ['mlp-softmax',  5, 0.0,  1.0, 0],
                ['mlp-softmax',  8, 0.0,  1.0, 0],
                ['mlp-softmax', 12, 0.0,  1.0, 0],
                ['mlp-softmax', 12, 0.0,  1.0, 100],
                ['mlp-softmax', 12, 1e-3, 1.0, 0],
                ['mlp-softmax', 12, 1e-4, 1.0, 0],
                ['mlp-softmax', 12, 1e-5, 1.0, 0],
                ['mlp-softmax', 12, 1e-6, 1.0, 0],
                ['mlp-softmax', 12, 1e-4, 1.0, 100],
                ['mlp-softmax', 12, 1e-4, 0.9, 0],
                ['mlp-softmax', 12, 1e-4, 0.8, 0],
                ['mlp-softmax', 12, 1e-4, 0.9, 100]])
    """
    """
    test_block('moons_separable', 12, range(1,2),
               [['mlp-sigmoid',  3, 0.0,  1.0, 0],
                ['mlp-sigmoid',  8, 0.0,  1.0, 0],
                ['mlp-sigmoid', 16, 0.0,  1.0, 0],
                ['mlp-sigmoid', 32, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 100],
                ['mlp-sigmoid', 64, 1e-3, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-5, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-6, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 100],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.6, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 100]])
    """
    """
    test_block_rc('moons_separable', 12, range(1,2),
               [['mlp-sigmoid',  3, 0.0,  1.0, 0],
                ['mlp-sigmoid',  8, 0.0,  1.0, 0],
                ['mlp-sigmoid', 16, 0.0,  1.0, 0],
                ['mlp-sigmoid', 32, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 100],
                ['mlp-sigmoid', 64, 1e-3, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-5, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-6, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 100],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.6, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 100],
                ['mlp-softmax',  3, 0.0,  1.0, 0],
                ['mlp-softmax',  8, 0.0,  1.0, 0],
                ['mlp-softmax', 16, 0.0,  1.0, 0],
                ['mlp-softmax', 32, 0.0,  1.0, 0],
                ['mlp-softmax', 64, 0.0,  1.0, 0],
                ['mlp-softmax', 64, 0.0,  1.0, 100],
                ['mlp-softmax', 64, 1e-3, 1.0, 0],
                ['mlp-softmax', 64, 1e-4, 1.0, 0],
                ['mlp-softmax', 64, 1e-5, 1.0, 0],
                ['mlp-softmax', 64, 1e-6, 1.0, 0],
                ['mlp-softmax', 64, 1e-4, 1.0, 100],
                ['mlp-softmax', 64, 1e-4, 0.9, 0],
                ['mlp-softmax', 64, 1e-4, 0.6, 0],
                ['mlp-softmax', 64, 1e-4, 0.9, 100]])
    """
    #test_block_RBF('moons_overlapping', 5, range(0,1), [4,8,12,16,20,24,32,36,40,44,48,52,56,60])
    #test_block_RBF('moons_separable', 12, range(0,1), [4,8,12,16,20,24,32,36,40,44,48,52,56,60])
    #wilcoxon_test()