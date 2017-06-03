'''
Created on May 31, 2017

@author: anton

Tests to generate results for Chapter 4 and Section 1 about generated classes
'''
from DataSet import DataSet
from MLP import MLP
from RBF import RBF
from graphics import plot_decision_regions, plot_confusion_matrix, \
    plot_2d_dataset, plot_multiclass_roc_curve, plot_binary_roc_curve
from data_utils import threshold_output, threshold_differential, threshold_ratio, \
    calc_roc_binary, calc_roc_multiclass, calc_roc_multiple
from sklearn import metrics

FIG_HALF_SIZE = 4.1
IMAGES_DIR = '/home/anton/Desktop/diploma_text/images/chapter_4_1/'

def test_mlp_sigmoid_thresholds(ds):
    mlp = MLP(0.01, [ds.num_features, 4, ds.num_classes], 'sigmoid', 'Adam')
    mlp.train(10000, ds.trn, ds.vld, 1, logging=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_threshold_output')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_threshold_differential')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_threshold_ratio')

def test_mlp_sotmax_thresholds(ds):
    mlp = MLP(0.01, [ds.num_features, 3, ds.num_classes], 'softmax', 'Adam')
    mlp.train(10000, ds.trn, ds.vld, 1, logging=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_threshold_output')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_threshold_differential')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_threshold_ratio')

def test_rbf_thresholds(ds):
    rbf = RBF(ds.num_features, 16, ds.num_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'rbf_threshold_output')
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'rbf_threshold_differential')
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'rbf_threshold_ratio')

def test_mlp_sigmoid_thresholds_noise_as_no_class(ds):
    mlp = MLP(0.01, [ds.num_features, 16, ds.num_classes], 'sigmoid', 'Adam')
    mlp.train(10000, ds.trn, ds.vld, 1, logging=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_t1_n1')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_t2_n1')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_t3_n1')

def test_mlp_sotmax_thresholds_noise_as_no_class(ds):
    mlp = MLP(0.01, [ds.num_features, 16, ds.num_classes], 'softmax', 'Adam')
    mlp.train(10000, ds.trn, ds.vld, 1, logging=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_t1_n1')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_t2_n1')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_t3_n1')

def test_mlp_sigmoid_thresholds_noise_as_a_class(ds):
    mlp = MLP(0.01, [ds.num_features, 16, ds.num_classes], 'sigmoid', 'Adam')
    mlp.train(10000, ds.trn, ds.vld, 1, logging=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_t1_n2')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_t2_n2')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_sigmoid_t3_n2')

def test_mlp_sotmax_thresholds_noise_as_a_class(ds):
    mlp = MLP(0.01, [ds.num_features, 16, ds.num_classes], 'softmax', 'Adam')
    mlp.train(10000, ds.trn, ds.vld, 1, logging=True)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_t1_n2')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_t2_n2')
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE),
                          IMAGES_DIR + 'mlp_softmax_t3_n2')

def test_thresholds():
    ds = DataSet(5)
    # test_mlp_sigmoid_thresholds(ds)
    # test_mlp_sotmax_thresholds(ds)
    test_rbf_thresholds(ds)
    
def test_thresholds_with_noise_as_no_class():
    ds = DataSet(5, add_noise=1, noise_output=0.0)
    test_mlp_sigmoid_thresholds_noise_as_no_class(ds)
    ds = DataSet(5, add_noise=1, noise_output=0.5)
    test_mlp_sotmax_thresholds_noise_as_no_class(ds)

def test_thresholds_with_noise_as_a_class():
    ds = DataSet(5, add_noise=2)
    test_mlp_sigmoid_thresholds_noise_as_a_class(ds)
    test_mlp_sotmax_thresholds_noise_as_a_class(ds)

def test_mlp_overfitting():
    ds = DataSet(5)
    mlp = MLP(0.01, [ds.num_features, 100, ds.num_classes], 'sigmoid', 'Adam')
    mlp.train(1000, ds.trn, None, 1, logging=True)
    
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_output,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE), None)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_differential,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE), None)
    plot_decision_regions(ds.tst.x, ds.tst.y, mlp, threshold_ratio,
                          (FIG_HALF_SIZE, FIG_HALF_SIZE), None)

def test_unit(ds, ds_name, clf_id, clf_name, n_hidden, show=True):
    if clf_id == 0:
        mlp = MLP(0.01, [ds.num_features, n_hidden, ds.num_classes],
                  'sigmoid', 'Adam')
    elif clf_id == 1:
        mlp = MLP(0.01, [ds.num_features, n_hidden, ds.num_classes],
                  'softmax', 'Adam')
    
    result = mlp.train(1000, ds.trn, ds.vld, 1, logging=True)
    print(result)
    
    outputs = mlp.predict_proba(ds.tst.x)
    y_pred = outputs.argmax(axis=1)
    y_true = ds.tst.y.argmax(axis=1)
    
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    print('Accuracy: {0:f}'.format(accuracy_score))
    
    #plot_2d_dataset(ds.trn.x, ds.trn.y.argmax(axis=1))
    
    savefig = 'tests/{0:s}/{1:s}/{2:s}_{0:s}_{1:s}_{3:03d}' \
                .format(ds_name, clf_name, '{:s}', n_hidden)
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
    
    fpr_bin, tpr_bin, auc_bin = calc_roc_binary(ds.tst.y, outputs)
    fpr_m, tpr_m, auc_m = calc_roc_multiple(ds.tst.y, outputs, ds.target_names)
    fpr_bin['m'] = fpr_m
    tpr_bin['m'] = tpr_m
    auc_bin['m'] = auc_m
    plot_binary_roc_curve(fpr_bin, tpr_bin, auc_bin,
                          savefig.format('roc_b'), show)
    
    fpr_mc, tpr_mc, auc_mc = calc_roc_multiclass(ds.tst.y, outputs,
                                                 ds.target_names)
    plot_multiclass_roc_curve(fpr_mc, tpr_mc, auc_mc, ds.target_names,
                              savefig.format('roc_m'), show)
    
    return '{:>8}, {:>16}, {:4d}, {:9f}, {:9f}, {:9f}, {:9f}, ' \
        '{:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}' \
        ''.format(ds_name, clf_name, n_hidden,
                  result[2], result[3], result[4], accuracy_score,
                  auc_bin[0], auc_bin[1], auc_bin[2], auc_m,
                  auc_mc[0], auc_mc[1], auc_mc['micro'], auc_mc['macro'])

def test_block():
    dataset_names = ['moons']
    dataset_ids = [5]
    
    classifier_names = ['mlp-sigmoid', 'mlp-softmax']
    classifier_ids = [0, 1]
    
    free_parameters = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    result = ''
    for ds_id, ds_name in zip(dataset_ids, dataset_names):
        ds = DataSet(ds_id)
        for clf_id, clf_name in zip(classifier_ids, classifier_names):
            for n_hidden in free_parameters:
                result += test_unit(ds, ds_name, clf_id, clf_name, n_hidden, 
                                    False) + '\n'
    print(result)

if __name__ == '__main__':
    # test_thresholds()
    # test_thresholds_with_noise_as_no_class()
    # test_thresholds_with_noise_as_a_class()
    # test_mlp_overfitting()
    test_block()