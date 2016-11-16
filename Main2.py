'''
Created on Nov 12, 2016

@author: anton
'''
from InputData import get_data
from DataUtils import split_in_tvt
from MLP import MLP

ds_x, ds_y = get_data(3)
trn_x, trn_y, vld_x, vld_y, tst_x, tst_y = split_in_tvt(ds_x, ds_y)

for learning_rate in [0.08, 0.1, 0.2]:
    for num_hidden in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        
        print("learning rate %g; hidden_num %d" % (learning_rate, num_hidden))
        filename = "saver/model_%g_%d.ckpt" % (learning_rate, num_hidden)
        
        mlp = MLP(learning_rate, trn_x.shape[1], num_hidden, trn_y.shape[1], activation_function='sigmoid')
        train_accuracy = mlp.train(100001, trn_x, trn_y, vld_x, vld_y, filename=filename, logging=False)
        print("Train accuracy %g" % (train_accuracy))
        
        test_accuracy = mlp.test(tst_x, tst_y, filename=filename, logging=True)
        print("+++++++    Test accuracy %g" % (test_accuracy))

        print()

print("The end.")