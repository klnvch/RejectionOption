'''
Created on Aug 27, 2017

@author: anton

Tests for Glass Identification Dataset
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_block_rbf, \
    test_unit_RBF


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

if __name__ == '__main__':
    
    # ds = DataSet(ds_id=3, split=[0.4, 0.1, 0.5])
    # test_unit_mlp(ds=ds, clf='mlp-sgm', rej=0, units=12,
    #              beta=0.0001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=5000, batch_size=32, print_step=100, show=True)
    
    # ds = DataSet(ds_id=3, split=[0.4, 0.1, 0.5])
    # test_unit_RBF(ds, 12, 4.0, True)
    
    test_block_mlp(3, 'glass', range(0, 1), None, [0.4, 0.1, 0.5], params_0)
