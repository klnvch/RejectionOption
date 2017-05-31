'''
Created on May 31, 2017

@author: anton

Tests to generate results for Chapter 4 and Section 1 about generated classes
'''
from DataSet import DataSet
from MLP import MLP
from RBF import RBF
from graphics import plot_decision_regions
from data_utils import threshold_output, threshold_differential, threshold_ratio

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
    rbf = RBF(ds.num_features, 64, ds.num_classes)
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
    test_mlp_sigmoid_thresholds(ds)
    #test_mlp_sotmax_thresholds(ds)
    #test_rbf_thresholds(ds)
    
def test_thresholds_with_noise_as_no_class():
    ds = DataSet(5, add_noise=1, noise_output=0.0)
    test_mlp_sigmoid_thresholds_noise_as_no_class(ds)
    ds = DataSet(5, add_noise=1, noise_output=0.5)
    test_mlp_sotmax_thresholds_noise_as_no_class(ds)

def test_thresholds_with_noise_as_a_class():
    ds = DataSet(5, add_noise=2)
    test_mlp_sigmoid_thresholds_noise_as_a_class(ds)
    test_mlp_sotmax_thresholds_noise_as_a_class(ds)

if __name__ == '__main__':
    #test_thresholds()
    #test_thresholds_with_noise_as_no_class()
    test_thresholds_with_noise_as_a_class()