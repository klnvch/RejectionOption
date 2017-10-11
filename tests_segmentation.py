'''
Created on Jun 24, 2017

@author: anton

Tests for Image Segmentation Dataset
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_block_rbf, \
    test_unit_RBF


# rej, clf, units, beta, dropout, es, targets, n_epochs, batch_size, print_step
params_0 = [
    
    [0, 'mlp-sgm', 3, 0.0, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # minimal size
    [0, 'mlp-sgm', 6, 0.0, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # 
    [0, 'mlp-sgm', 32, 0.0, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # overfitting
    [0, 'mlp-sgm', 32, 0.0, 1.0, 10, (0.0, 1.0), 3000, 32, 100],  # early stop
    [0, 'mlp-sgm', 32, 1e-3, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # high penalty
    [0, 'mlp-sgm', 32, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  #  mid penalty
    [0, 'mlp-sgm', 32, 1e-5, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  #  low penalty
    [0, 'mlp-sgm', 32, 1e-4, 1.0, 0, (0.1, 0.9), 3000, 32, 100],  # targets
    [0, 'mlp-sgm', 32, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],  # penalty & es
    [0, 'mlp-sgm', 32, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],  # penalt & drop
    [0, 'mlp-sgm', 32, 1e-4, 0.5, 10, (0.0, 1.0), 3000, 32, 100],  # everything
    
    [0, 'mlp-sft', 3, 0.0, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # minimal size
    [0, 'mlp-sft', 6, 0.0, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # 
    [0, 'mlp-sft', 32, 0.0, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # overfitting
    [0, 'mlp-sft', 32, 0.0, 1.0, 10, (0.0, 1.0), 3000, 32, 100],  # early stop
    [0, 'mlp-sft', 32, 1e-3, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # high penalty
    [0, 'mlp-sft', 32, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  #  mid penalty
    [0, 'mlp-sft', 32, 1e-5, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  #  low penalty
    [0, 'mlp-sft', 32, 1e-4, 1.0, 0, (0.1, 0.9), 3000, 32, 100],  # targets
    [0, 'mlp-sft', 32, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],  # penalty & es
    [0, 'mlp-sft', 32, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],  # penalt & drop
    [0, 'mlp-sft', 32, 1e-4, 0.5, 10, (0.0, 1.0), 3000, 32, 100],  # everything
    
    [0, 'rbf', 32, 1e-3, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [0, 'rbf', 32, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [0, 'rbf', 32, 1e-5, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    
    [0, 'rbf-reg', 32, 1e-3, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [0, 'rbf-reg', 32, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [0, 'rbf-reg', 32, 1e-5, 1.0, 0, (0.0, 1.0), 3000, 32, 100]
    
    ]

params_1 = [
    
    [1, 'mlp-sgm', 64, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # penalty
    [1, 'mlp-sgm', 64, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],  # es
    [1, 'mlp-sgm', 64, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],  # dropout
    
    [1, 'mlp-sgm-2', 64, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],  # penalty
    [1, 'mlp-sgm-2', 64, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],  # es
    [1, 'mlp-sgm-2', 64, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],  # dropout
    
    [2, 'mlp-sgm', 64, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sgm', 64, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sgm', 64, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],
    
    [2, 'mlp-sgm-2', 64, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sgm-2', 64, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sgm-2', 64, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],
    
    [2, 'mlp-sft', 64, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sft', 64, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sft', 64, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],
    
    [2, 'mlp-sft-2', 64, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sft-2', 64, 1e-4, 1.0, 10, (0.0, 1.0), 3000, 32, 100],
    [2, 'mlp-sft-2', 64, 1e-4, 0.5, 0, (0.0, 1.0), 3000, 32, 100],
    
    [3, 'rbf', 32, 1e-3, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [3, 'rbf', 32, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [3, 'rbf', 32, 1e-5, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    
    [3, 'rbf-reg', 32, 1e-3, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [3, 'rbf-reg', 32, 1e-4, 1.0, 0, (0.0, 1.0), 3000, 32, 100],
    [3, 'rbf-reg', 32, 1e-5, 1.0, 0, (0.0, 1.0), 3000, 32, 100]
    
    ]

params_2 = [
    [4.0, 12],
    [8.0, 12],
    [4.0, 16],
    [8.0, 16],
    ]

if __name__ == '__main__':
    # ds = DataSet(4, split=[0.4, 0.1, 0.5])
    # test_unit_mlp(ds=ds, clf='mlp-sft', rej=0, units=32,
    #             beta=0.00001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #             n_epochs=1000, batch_size=32, print_step=100, show=True)
    
    test_block_mlp(4, 'segmentation', range(0, 1), 1000, [0.4, 0.1, 0.5],
                   params_0)
