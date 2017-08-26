'''
Created on Jun 24, 2017

@author: anton

Tests for Iris Dataset
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_unit_RBF,\
    test_block_rbf

params_1 = [
    ['mlp-sgm', 8, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 0.9, 10, (0.0, 1.0), 5000]
    ]

params_2 = [
    ['mlp-sgm', 8, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sgm', 8, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-sft', 8, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sft', 8, 1e-4, 0.9, 10, (0.0, 1.0), 5000]
    ]

params_3 = [
    ['rbf', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 8, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    
    ['rbf-reg', 8, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 8, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 8, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 8, 1e-6, 1.0, 0,  (0.0, 1.0), 5000],
    ]

params_4 = [
    [1, 6],
    [1, 8],
    [1, 12],
    [1, 16]
    ]

if __name__ == '__main__':
    #ds = DataSet(2, split=[0.4, 0.1, 0.5])
    #test_unit_mlp(ds=ds, clf='mlp-sgm', rej=0, units=12,
    #              beta=0.001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=5000, show=True)
    
    #ds = DataSet(2, split=[0.4, 0.1, 0.5], add_noise=1)
    #test_unit_mlp(ds=ds, clf='mlp-sgm', rej=1, units=12,
    #              beta=0.001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=5000, show=True)
    
    #ds = DataSet(2, split=[0.4, 0.1, 0.5], add_noise=2)
    #test_unit_mlp(ds=ds, clf='mlp-sgm', rej=2, units=12,
    #              beta=0.001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=5000, show=True)
    
    #ds = DataSet(ds_id=2, split=[0.4, 0.1, 0.5], add_noise=3)
    #test_unit_RBF(ds, 8, 0.0001, True)
    
    # test outliers rejection performance
    # mlp-sgm trained with noise as no class
    # mlp trained with outliers as a class 
    # rbf
    
    #test_block_mlp(2, 'iris', 1, range(0,1), None, [0.4, 0.1, 0.5], params_1)
    #test_block_mlp(2, 'iris', 2, range(0,1), None, [0.4, 0.1, 0.5], params_2)
    #test_block_mlp(2, 'iris', 3, range(0,1), None, [0.4, 0.1, 0.5], params_3)
    
    test_block_rbf('iris', 2, range(0,1), None, [0.4, 0.1, 0.5], params_4)