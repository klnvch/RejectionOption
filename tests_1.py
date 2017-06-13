'''
Created on May 31, 2017

@author: anton

Tests to generate results for Chapter 4 and Section 1 about generated classes
'''
from DataSet import DataSet
from MLP import MLP
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

def test_mlp_reject_class():
    #ds = DataSet(5, 1000, [0.2, 0.8])
    #ds = DataSet(5, 1000, [0.2, 0.2, 0.6])
    ds = DataSet(5, 600, [0.34, 0.66], add_noise=2)
    mlp = MLP(0.01, [ds.n_features, 20, ds.n_classes], 'softmax', 'Adam')
    result = mlp.train(20000, ds.trn, ds.vld, 20, False)
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
    ds = DataSet(12, 600, [0.34, 0.66], add_noise=1)
    mlp = MLP(0.01, [ds.n_features, 8, ds.n_classes], 'sigmoid', 'Adam')
    result = mlp.train(20000, ds.trn, ds.vld, 20, False)
    print(result)
    
    score = mlp.score(ds.tst)
    print('Test accuracy: {0:f}'.format(score))
    
    outputs = mlp.predict_proba(ds.tst.x)
    outputs_outl = mlp.predict_proba(ds.outliers)
    
    plot_2d_dataset(ds.tst.x, ds.tst.y)
    plot_confusion_matrix(outputs, ds.tst.y, ds.target_names)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_output)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_diff)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, thr_ratio)
    
    scores_s = [score_outp, score_diff, score_rati]
    curves_s = roc_s_thr(ds.tst.y, outputs, outputs_outl, scores_s)
    
    scores_m = [score_outp_m]
    curves_m = roc_m_thr(ds.n_classes, ds.tst.y, outputs, outputs_outl, scores_m)
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]))
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names, outputs_outl)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names)

def test_unit(ds, ds_name, attempt, clf_id, clf_name, n_hidden, show=True):
    if clf_id == 0:
        mlp = MLP(0.01, [ds.n_features, n_hidden, ds.n_classes],
                  'sigmoid', 'Adam')
    elif clf_id == 1:
        mlp = MLP(0.01, [ds.n_features, n_hidden, ds.n_classes],
                  'softmax', 'Adam')
    result = mlp.train(20000, ds.trn, ds.vld, 20, False)
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
    
    plot_roc_curves(np.concatenate([curves_s, curves_m]))
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names,
                                                 outputs_outl)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names,
                              savefig.format('roc_m'), show)
    
    return '{:>8}, {:4d}, {:>16}, {:4d}, {:9f}, {:9f}, {:9f}, {:9f}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}' \
        ''.format(ds_name, attempt, clf_name, n_hidden,
                  result[2], result[3], result[4], score,
                  curves_s[0][2], curves_s[1][2], curves_s[2][2], curves_s[3][2],
                  auc_mc[0], auc_mc[1], auc_mc['micro'], auc_mc['macro'])

def test_block():
    dataset_names = ['moons']
    #dataset_ids = [5]
    dataset_ids = [12]
    
    classifier_names = ['mlp-sigmoid']
    classifier_ids = [0]
    
    #free_parameters = [1,2,3,4,5,6,7,8,9]
    free_parameters = [3,5,8,12,17,23,30,38]
    
    for ds_id, ds_name in zip(dataset_ids, dataset_names):
        filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
        
        for attempt in range(15,21):
            #ds = DataSet(ds_id, 1000, [0.2, 0.8])
            ds = DataSet(ds_id, 600, [0.34, 0.66], add_noise=1)
            for clf_id, clf_name in zip(classifier_ids, classifier_names):
                for n_hidden in free_parameters:
                    msg = test_unit(ds, ds_name, attempt, clf_id, clf_name,
                                    n_hidden, False)
                    with open(filename, 'a+') as f:
                        print(msg, file=f)

def wilcoxon_test():
    df = pd.read_csv('tests/moons/run_1496670425.csv')
    print(df.keys())
    
    mot = [df.loc[((df['Hidden units'] == i) & (df['Classifier'].str.strip() == 'mlp-softmax')), 'Multiple output threshold'].values for i in range(1,13)]
    
    A = [[scipy.stats.wilcoxon(mot[i], mot[j]).pvalue for i in range(12)] for j in range(12)]
    
    print('\n'.join([''.join(['{:4f} '.format(item) for item in row]) for row in A]))

if __name__ == '__main__':
    #test_mlp()
    test_mlp_reject_class()
    #test_block()
    #wilcoxon_test()