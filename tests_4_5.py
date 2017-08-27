'''
Created on Aug 27, 2017

@author: anton

Tests for Glass Identification Dataset
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_block_rbf,\
    test_unit_RBF

params_1 = [
    ['mlp-sgm', 8, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-sgm', 12, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 12, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-sgm', 16, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 16, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-sft', 8, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-sft', 12, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 12, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 12, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 12, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-sft', 16, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 16, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 16, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 16, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 16, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 16, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 16, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 16, 1e-4, 0.9, 10, (0.0, 1.0), 5000]
    ]

params_2 = [
    ['rbf', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 8, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    
    ['rbf', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    
    ['rbf', 16, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 16, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 16, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 16, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    
    ['rbf-reg', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 8, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    
    ['rbf-reg', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    
    ['rbf-reg', 16, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 16, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 16, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 16, 1e-6, 1.0, 0,  (0.0, 1.0), 5000]
    ]

params_2 = [
    [1, 6],
    [1, 8],
    [1, 12],
    [1, 16],
    [1, 20],
    [1, 24]
    ]

if __name__ == '__main__':
    ds = DataSet(3, split=[0.4, 0.1, 0.5], add_noise=0)
    test_unit_mlp(ds=ds, clf='mlp-sgm', rej=0, units=16,
                  beta=0.001, dropout=1.0, es=0, targets=(0.0, 1.0),
                  n_epochs=5000, show=True)
    
    #ds = DataSet(ds_id=3, split=[0.4, 0.1, 0.5], add_noise=0)
    test_unit_RBF(ds, 16, 1.0, True)
    
    # misclassifiactions
    #test_block_mlp(3, 'glass', 0, range(0,1), None, [0.4, 0.1, 0.5], params_1)
    #test_block_mlp(3, 'glass', 0, range(0,1), None, [0.4, 0.1, 0.5], params_2)
    #test_block_rbf('glass', 3, range(0,1), None, [0.4, 0.1, 0.5], params_2)