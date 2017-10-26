'''
Created on May 31, 2017

@author: anton

Tests for Generated Datasets
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_unit_RBF, \
    test_block_rbf


# rej, clf, units, beta, dropout, es, targets, n_epochs, batch_size, print_step
params_0 = [
    
    [0, 'mlp-sgm', 3, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # minimal size
    [0, 'mlp-sgm', 6, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # 
    [0, 'mlp-sgm', 12, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # overfitting
    [0, 'mlp-sgm', 12, 0.0, 1.0, 10, (0.0, 1.0), 5000, 32, 100],  # early stop
    [0, 'mlp-sgm', 12, 1e-3, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # high penalty
    [0, 'mlp-sgm', 12, 1e-4, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  #  mid penalty
    [0, 'mlp-sgm', 12, 1e-5, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  #  low penalty
    [0, 'mlp-sgm', 12, 1e-4, 1.0, 0, (0.1, 0.9), 5000, 32, 100],  # targets
    [0, 'mlp-sgm', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000, 32, 100],  # penalty & es
    [0, 'mlp-sgm', 12, 1e-4, 0.5, 0, (0.0, 1.0), 5000, 32, 100],  # penalt & drop
    [0, 'mlp-sgm', 12, 1e-4, 0.5, 10, (0.0, 1.0), 5000, 32, 100],  # everything
    
    [0, 'mlp-sft', 3, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # minimal size
    [0, 'mlp-sft', 6, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # 
    [0, 'mlp-sft', 12, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # overfitting
    [0, 'mlp-sft', 12, 0.0, 1.0, 10, (0.0, 1.0), 5000, 32, 100],  # early stop
    [0, 'mlp-sft', 12, 1e-3, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # high penalty
    [0, 'mlp-sft', 12, 1e-4, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  #  mid penalty
    [0, 'mlp-sft', 12, 1e-5, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  #  low penalty
    [0, 'mlp-sft', 12, 1e-4, 1.0, 0, (0.1, 0.9), 5000, 32, 100],  # targets
    [0, 'mlp-sft', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000, 32, 100],  # penalty & es
    [0, 'mlp-sft', 12, 1e-4, 0.5, 0, (0.0, 1.0), 5000, 32, 100],  # penalt & drop
    [0, 'mlp-sft', 12, 1e-4, 0.5, 10, (0.0, 1.0), 5000, 32, 100],  # everything
    
    [0, 'rbf', 12, 1e-3, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [0, 'rbf', 12, 1e-4, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [0, 'rbf', 12, 1e-5, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    
    [0, 'rbf-reg', 12, 1e-3, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [0, 'rbf-reg', 12, 1e-4, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [0, 'rbf-reg', 12, 1e-5, 1.0, 0, (0.0, 1.0), 10000, 32, 100]
    
    ]

params_1 = [
    
    [1, 'mlp-sgm', 12, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # low
    [1, 'mlp-sgm', 32, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # high
    [1, 'mlp-sgm', 32, 1e-4, 1.0, 0, (0.0, 1.0), 5000, 32, 100],  # penalty
    [1, 'mlp-sgm', 32, 1e-4, 1.0, 0, (0.1, 0.9), 5000, 32, 100],  # targets
    [1, 'mlp-sgm', 32, 1e-4, 1.0, 10, (0.0, 1.0), 5000, 32, 100],  # es
    [1, 'mlp-sgm', 32, 1e-4, 0.5, 0, (0.0, 1.0), 5000, 32, 100],  # dropout
    
    [2, 'mlp-sgm', 12, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sgm', 32, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sgm', 32, 1e-4, 1.0, 0, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sgm', 32, 1e-4, 1.0, 0, (0.1, 0.9), 5000, 32, 100],
    [2, 'mlp-sgm', 32, 1e-4, 1.0, 10, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sgm', 32, 1e-4, 0.5, 0, (0.0, 1.0), 5000, 32, 100],
    
    [2, 'mlp-sft', 12, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sft', 32, 0.0, 1.0, 0, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sft', 32, 1e-4, 1.0, 0, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sft', 32, 1e-4, 1.0, 0, (0.1, 0.9), 5000, 32, 100],
    [2, 'mlp-sft', 32, 1e-4, 1.0, 10, (0.0, 1.0), 5000, 32, 100],
    [2, 'mlp-sft', 32, 1e-4, 0.5, 0, (0.0, 1.0), 5000, 32, 100],
    
    [3, 'rbf', 12, 1e-3, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [3, 'rbf', 12, 1e-4, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [3, 'rbf', 12, 1e-5, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    
    [3, 'rbf-reg', 12, 1e-3, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [3, 'rbf-reg', 12, 1e-4, 1.0, 0, (0.0, 1.0), 10000, 32, 100],
    [3, 'rbf-reg', 12, 1e-5, 1.0, 0, (0.0, 1.0), 10000, 32, 100]
    
    ]

params_2 = [
    [4.0, 12],
    [8.0, 12],
    [4.0, 16],
    [8.0, 16],
    ]

if __name__ == '__main__':
    #  misclassifications
    
    ds = DataSet.load_data(ds_id=9, n_samples=1000, split=[0.1, 0.1, 0.8])
    test_unit_mlp(ds=ds, clf='mlp-sft', rej=0, units=24,
                  beta=0.0001, dropout=1.0, es=0, targets=(0.0, 1.0),
                  n_epochs=1000, batch_size=32, print_step=100, show=True)
    
    # ds = DataSet(ds_id=12, n_samples=1000, split=[0.1, 0.1, 0.8])
    # test_unit_RBF(ds, 12, 4.0, True)
    
    # blocks misclassifications
    # test_block_mlp(5, 'moons_overlapping', range(0,1), 1000, [0.1, 0.1, 0.8],
    #               params_0)
    # test_block_mlp(9, 'multiclass', range(0,1), 1000, [0.1, 0.1, 0.8],
    #               params_0, random_state=47)
    # blocks outliers
    # test_block_mlp(12, 'moons_separable', range(0,1), 1000, [0.1, 0.1, 0.8],
    #               params_1)
    # test_block_mlp(9, 'multiclass', range(0, 1), 1000, [0.1, 0.1, 0.8],
    #               params_1, random_state=47)
    # test_block_rbf(12, 'moons_separable', range(0,1), 1000, [0.1, 0.1, 0.8],
    #               params_2)
    # test_block_rbf(9, 'multiclass', range(0, 1), 1000, [0.1, 0.1, 0.8],
    #               params_2, random_state=47)
