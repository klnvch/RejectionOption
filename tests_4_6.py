'''
Created on Aug 29, 2017

@author: anton
Tests for MNIST dataset
'''
from DataSet import DataSet
from tests_common import test_unit_mlp, test_block_mlp, test_block_rbf,\
    test_unit_RBF

params_0 = [
    ['mlp-sgm', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 500, 512, 10],
    ['mlp-sgm', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 500, 512, 10],
    ['mlp-sgm', 256, 0.00001,  1.0, 0,  (0.0, 1.0), 500, 512, 10],
    
    ['mlp-sft', 64, 0.00001,  1.0, 0,  (0.0, 1.0), 500, 512, 10],
    ['mlp-sft', 128, 0.00001,  1.0, 0,  (0.0, 1.0), 500, 512, 10],
    ['mlp-sft', 256, 0.00001,  1.0, 0,  (0.0, 1.0), 500, 512, 10],
    
    ['conv-sgm', 64, 0.0,  0.8, 0,  (0.0, 1.0), 5, 100, 1],
    ['conv-sgm', 128, 0.0,  0.8, 0,  (0.0, 1.0), 5, 100, 1],
    ['conv-sgm', 256, 0.0,  0.8, 0,  (0.0, 1.0), 5, 100, 1],
    
    ['conv-sft', 64, 0.0,  0.8, 0,  (0.0, 1.0), 5, 100, 1],
    ['conv-sft', 128, 0.0,  0.8, 0,  (0.0, 1.0), 5, 100, 1],
    ['conv-sft', 256, 0.0,  0.8, 0,  (0.0, 1.0), 5, 100, 1]
    ]

if __name__ == '__main__':
    ds = DataSet(14, add_noise=8)
    test_unit_mlp(ds=ds, clf='conv-sgm', rej=8, units=128,
                  beta=0.0, dropout=0.8, es=0, targets=(0.0, 1.0),
                  n_epochs=5, batch_size=128, print_step=1, show=True)
    
    #test_block_mlp(14, 'mnist', 0, range(0,1), None, None, params_0)
    #test_block_mlp(14, 'mnist', 3, range(0,1), None, None, params_0)