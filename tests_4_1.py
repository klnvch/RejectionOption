'''
Created on May 31, 2017

@author: anton

Tests to generate results for Chapter 4 and Section 1 about generated classes
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp

# rej, clf, noisi_size, units, beta, dropout, es, targets, epochs

params_0 = [
    
    ['mlp-sigmoid',  3, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid',  6, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 1e-4, 1.0, 0,  (0.1, 0.9), 5000],
    ['mlp-sigmoid', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-sigmoid', 12, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['mlp-softmax',  3, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax',  6, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 1e-4, 1.0, 0,  (0.1, 0.9), 5000],
    ['mlp-softmax', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['mlp-softmax', 12, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['rbf',  3, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['rbf',  6, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['rbf', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-4, 1.0, 0,  (0.1, 0.9), 5000],
    ['rbf', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['rbf', 12, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['rbf', 12, 1e-4, 0.9, 10, (0.0, 1.0), 5000],
    
    ['rbf-reg',  3, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg',  6, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 0.0,  1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 0.0,  1.0, 10, (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-3, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-4, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-5, 1.0, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-4, 1.0, 0,  (0.1, 0.9), 5000],
    ['rbf-reg', 12, 1e-4, 1.0, 10, (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-4, 0.9, 0,  (0.0, 1.0), 5000],
    ['rbf-reg', 12, 1e-4, 0.9, 10, (0.0, 1.0), 5000]
    
    ]

if __name__ == '__main__':
    #ds = DataSet(ds_id=5, size=1000, split=[0.1, 0.1, 0.8], add_noise=0)
    #test_unit_mlp(ds=ds, clf='mlp-softmax', rej=0, units=12,
    #              beta=0.0001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=5000, show=True)
    
    #ds = DataSet(ds_id=5, size=1000, split=[0.1, 0.1, 0.8], add_noise=0)
    #test_unit_mlp(ds=ds, clf='rbf-reg', rej=0, units=12,
    #              beta=0.001, dropout=1.0, es=0, targets=(0.0, 1.0),
    #              n_epochs=5000, show=True)
    
    test_block_mlp(5, 'moons_overlapping', 0, range(0,1), params_0)
    """
    test_block('moons_separable', 12, range(1,2),
               [['mlp-sigmoid',  3, 0.0,  1.0, 0],
                ['mlp-sigmoid',  8, 0.0,  1.0, 0],
                ['mlp-sigmoid', 16, 0.0,  1.0, 0],
                ['mlp-sigmoid', 32, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 100],
                ['mlp-sigmoid', 64, 1e-3, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-5, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-6, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 100],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.6, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 100]])
    """
    """
    test_block_rc('moons_separable', 12, range(1,2),
               [['mlp-sigmoid',  3, 0.0,  1.0, 0],
                ['mlp-sigmoid',  8, 0.0,  1.0, 0],
                ['mlp-sigmoid', 16, 0.0,  1.0, 0],
                ['mlp-sigmoid', 32, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 0],
                ['mlp-sigmoid', 64, 0.0,  1.0, 100],
                ['mlp-sigmoid', 64, 1e-3, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-5, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-6, 1.0, 0],
                ['mlp-sigmoid', 64, 1e-4, 1.0, 100],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.6, 0],
                ['mlp-sigmoid', 64, 1e-4, 0.9, 100],
                ['mlp-softmax',  3, 0.0,  1.0, 0],
                ['mlp-softmax',  8, 0.0,  1.0, 0],
                ['mlp-softmax', 16, 0.0,  1.0, 0],
                ['mlp-softmax', 32, 0.0,  1.0, 0],
                ['mlp-softmax', 64, 0.0,  1.0, 0],
                ['mlp-softmax', 64, 0.0,  1.0, 100],
                ['mlp-softmax', 64, 1e-3, 1.0, 0],
                ['mlp-softmax', 64, 1e-4, 1.0, 0],
                ['mlp-softmax', 64, 1e-5, 1.0, 0],
                ['mlp-softmax', 64, 1e-6, 1.0, 0],
                ['mlp-softmax', 64, 1e-4, 1.0, 100],
                ['mlp-softmax', 64, 1e-4, 0.9, 0],
                ['mlp-softmax', 64, 1e-4, 0.6, 0],
                ['mlp-softmax', 64, 1e-4, 0.9, 100]])
    """
    #test_block_RBF('moons_overlapping', 5, range(0,1), [4,8,12,16,20,24,32,36,40,44,48,52,56,60])
    #test_block_RBF('moons_separable', 12, range(0,1), [4,8,12,16,20,24,32,36,40,44,48,52,56,60])