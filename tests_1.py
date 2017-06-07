'''
Created on May 31, 2017

@author: anton

Tests to generate results for Chapter 4 and Section 1 about generated classes
'''
from DataSet import DataSet
from MLP import MLP
from graphics import plot_decision_regions, plot_confusion_matrix, \
    plot_2d_dataset, plot_multiclass_roc_curve, plot_binary_roc_curve
from data_utils import threshold_output, threshold_differential, threshold_ratio, \
    calc_roc_binary, calc_roc_multiclass, calc_roc_multiple
from sklearn import metrics
import time
import pandas as pd
import scipy.stats

FIG_HALF_SIZE = 4.1

def test_mlp():
    #ds = DataSet(5, 1000, [0.2, 0.8])
    ds = DataSet(12, 600, [0.34, 0.66], add_noise=1)
    mlp = MLP(0.01, [ds.n_features, 60, ds.n_classes], 'sigmoid', 'Adam')
    mlp.train(20000, ds.trn, None, 20, logging=False)
    
    outputs = mlp.predict_proba(ds.tst.x)
    y_pred = outputs.argmax(axis=1)
    y_true = ds.tst.y.argmax(axis=1)
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    print('Accuracy: {0:f}'.format(accuracy_score))
    
    outputs_outl = mlp.predict_proba(ds.outliers)
    
    plot_2d_dataset(ds.tst.x, ds.tst.y.argmax(axis=1))
    plot_confusion_matrix(y_true, y_pred, ds.target_names)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio)
    
    fpr_bin, tpr_bin, auc_bin = calc_roc_binary(ds.tst.y, outputs, outputs_outl)
    fpr_m, tpr_m, auc_m = calc_roc_multiple(ds.tst.y, outputs, ds.target_names,
                                            outputs_outl)
    fpr_bin['m'] = fpr_m
    tpr_bin['m'] = tpr_m
    auc_bin['m'] = auc_m
    plot_binary_roc_curve(fpr_bin, tpr_bin, auc_bin)
    
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
    
    result = mlp.train(20000, ds.trn, None, 20, logging=False)
    print(result)
    
    outputs = mlp.predict_proba(ds.tst.x)
    y_pred = outputs.argmax(axis=1)
    y_true = ds.tst.y.argmax(axis=1)
    
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    print('Accuracy: {0:f}'.format(accuracy_score))
    
    outputs_outl = mlp.predict_proba(ds.outliers)
    
    #file name example: boundaries_0_moons_000_mlp-sigmoid_006 
    savefig = 'tests/{0:s}/{2:s}/{3:s}_{0:s}_{1:03d}_{2:s}_{4:03d}' \
                .format(ds_name, attempt, clf_name, '{:s}', n_hidden)
    plot_confusion_matrix(y_true, y_pred, ds.target_names,
                          savefig.format('confusion_matrix'), show)
    
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          savefig.format('boundaries_0'), show)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          savefig.format('boundaries_1'), show)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          savefig.format('boundaries_2'), show)
    
    fpr_bin, tpr_bin, auc_bin = calc_roc_binary(ds.tst.y, outputs, outputs_outl)
    fpr_m, tpr_m, auc_m = calc_roc_multiple(ds.tst.y, outputs, ds.target_names,
                                            outputs_outl)
    fpr_bin['m'] = fpr_m
    tpr_bin['m'] = tpr_m
    auc_bin['m'] = auc_m
    plot_binary_roc_curve(fpr_bin, tpr_bin, auc_bin,
                          savefig.format('roc_b'), show)
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names,
                                                 outputs_outl)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names,
                              savefig.format('roc_m'), show)
    
    return '{:>8}, {:4d}, {:>16}, {:4d}, {:9f}, {:9f}, {:9f}, {:9f}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}' \
        ''.format(ds_name, attempt, clf_name, n_hidden,
                  result[2], result[3], result[4], accuracy_score,
                  auc_bin[0], auc_bin[1], auc_bin[2], auc_m,
                  auc_mc[0], auc_mc[1], auc_mc['micro'], auc_mc['macro'])

def test_block():
    dataset_names = ['moons']
    #dataset_ids = [5]
    dataset_ids = [12]
    
    classifier_names = ['mlp-sigmoid']
    classifier_ids = [0]
    
    #free_parameters = [1,2,3,4,5,6,7,8,9]
    free_parameters = [3,5,7,9,11,13,15,17,19]
    
    for ds_id, ds_name in zip(dataset_ids, dataset_names):
        filename = 'tests/{:s}/run_{:d}.csv'.format(ds_name, int(time.time()))
        
        for attempt in range(6,7):
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
    test_mlp()
    #test_block()
    #wilcoxon_test()