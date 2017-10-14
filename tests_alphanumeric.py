'''
Created on Jun 24, 2017

@author: anton

Tests for Alphanumeric Dataset
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_block_rbf, \
    test_unit_RBF


# rej, clf, units, beta, dropout, es, targets, n_epochs, batch_size, print_step
params_0 = [
    
    [0, 'mlp-sgm', 64, 0.0, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # minimal size
    [0, 'mlp-sgm', 256, 0.0, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # 
    [0, 'mlp-sgm', 128, 0.0, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # overfitting
    [0, 'mlp-sgm', 128, 0.0, 1.0, 10, (0.0, 1.0), 1000, 32, 100],  # early stop
    [0, 'mlp-sgm', 128, 1e-3, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # high penalty
    [0, 'mlp-sgm', 128, 1e-4, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  #  mid penalty
    [0, 'mlp-sgm', 128, 1e-5, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  #  low penalty
    [0, 'mlp-sgm', 128, 1e-4, 1.0, 0, (0.1, 0.9), 1000, 32, 100],  # targets
    [0, 'mlp-sgm', 128, 1e-4, 1.0, 10, (0.0, 1.0), 1000, 32, 100],  # penalty & es
    [0, 'mlp-sgm', 128, 1e-4, 0.5, 0, (0.0, 1.0), 1000, 32, 100],  # penalt & drop
    [0, 'mlp-sgm', 128, 1e-4, 0.5, 10, (0.0, 1.0), 1000, 32, 100],  # everything
     
    [0, 'mlp-sft', 64, 0.0, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # minimal size
    [0, 'mlp-sft', 256, 0.0, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # 
    [0, 'mlp-sft', 128, 0.0, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # overfitting
    [0, 'mlp-sft', 128, 0.0, 1.0, 10, (0.0, 1.0), 1000, 32, 100],  # early stop
    [0, 'mlp-sft', 128, 1e-3, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  # high penalty
    [0, 'mlp-sft', 128, 1e-4, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  #  mid penalty
    [0, 'mlp-sft', 128, 1e-5, 1.0, 0, (0.0, 1.0), 1000, 32, 100],  #  low penalty
    [0, 'mlp-sft', 128, 1e-4, 1.0, 0, (0.1, 0.9), 1000, 32, 100],  # targets
    [0, 'mlp-sft', 128, 1e-4, 1.0, 10, (0.0, 1.0), 1000, 32, 100],  # penalty & es
    [0, 'mlp-sft', 128, 1e-4, 0.5, 0, (0.0, 1.0), 1000, 32, 100],  # penalt & drop
    [0, 'mlp-sft', 128, 1e-4, 0.5, 10, (0.0, 1.0), 1000, 32, 100],  # everything
    
    [0, 'rbf', 128, 1e-3, 1.0, 0, (0.0, 1.0), 200, 32, 10],
    [0, 'rbf', 128, 1e-4, 1.0, 0, (0.0, 1.0), 200, 32, 10],
    [0, 'rbf', 128, 1e-5, 1.0, 0, (0.0, 1.0), 200, 32, 10],
    
    [0, 'rbf-reg', 128, 1e-3, 1.0, 0, (0.0, 1.0), 200, 32, 10],
    [0, 'rbf-reg', 128, 1e-4, 1.0, 0, (0.0, 1.0), 200, 32, 10],
    [0, 'rbf-reg', 128, 1e-5, 1.0, 0, (0.0, 1.0), 200, 32, 10]
    
    ]

params_1 = [
    [1, 64],
    [1, 128],
    [1, 256],
    [1, 512]
    ]

params_2 = [
    # ['mlp-sgm', 128, 0.00001,  1.0, 0, (0.0, 1.0), 1000, 128, 10],
    # ['mlp-sft', 128, 0.00001,  1.0, 0, (0.0, 1.0), 1000, 128, 10],
    # ['mlp-sgm-2', 64, 0.00001,  1.0, 0, (0.0, 1.0), 1000, 128, 10],
    # ['mlp-sft-2', 64, 0.00001,  1.0, 0, (0.0, 1.0), 1000, 128, 10],
    
    ['rbf', 128, 0.0001, 1.0, 0, (0.0, 1.0), 300, 128, 10],
    ['rbf', 128, 0.00001, 1.0, 0, (0.0, 1.0), 300, 128, 10],
    ['rbf', 128, 0.000001, 1.0, 0, (0.0, 1.0), 300, 128, 10],
    
    # ['rbf-reg', 128, 0.0000001,   1.0, 0,  (0.0, 1.0), 300, 128, 10],
    # ['rbf-reg', 128, 0.00000001,  1.0, 0,  (0.0, 1.0), 300, 128, 10],
    # ['rbf-reg', 128, 0.000000001, 1.0, 0,  (0.0, 1.0), 300, 128, 10]
    ]

params_3 = [
    ['mlp-sgm', 256, 0.00001, 1.0, 0, (0.0, 1.0), 1000, 128, 10],
    ['mlp-sgm-2', 128, 0.00001, 1.0, 0, (0.0, 1.0), 1000, 128, 10]
    ]

params_4 = [
    ['mlp-sgm', 256, 0.00001, 1.0, 0, (0.0, 1.0), 1000, 128, 10],
    ['mlp-sft', 256, 0.00001, 1.0, 0, (0.0, 1.0), 1000, 128, 10],
    ['mlp-sgm-2', 128, 0.00001, 1.0, 0, (0.0, 1.0), 1000, 128, 10],
    ['mlp-sft-2', 128, 0.00001, 1.0, 0, (0.0, 1.0), 1000, 128, 10]
    ]

if __name__ == '__main__':
    # wicoxon_test('tests/alphanumeric/run_0.csv', params_0)
    # wicoxon_test('tests/alphanumeric/run_1.csv', params_1)
    
    ds = DataSet(13)
    # test_unit_mlp(ds=ds, clf='mlp-sgm', rej=0, units=128,
    #              beta=1e-3, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=100, batch_size=32, print_step=10, show=True)
    test_unit_mlp(ds=ds, clf='rbf', rej=0, units=128,
                  beta=1e-4, dropout=1.0, es=0, targets=(0.0, 1.0),
                  n_epochs=500, batch_size=128, print_step=10, show=True)
    
    # test_block_mlp(13, 'alphanumeric', range(0, 1), None, None, params_0)
