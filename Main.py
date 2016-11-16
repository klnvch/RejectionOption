'''
Created on Oct 11, 2016

@author: anton
'''

from InputData import get_data
from DataUtils import split
from MLP import MLP
    

ds_values, ds_labels = get_data(3)


for learning_rate in [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4]:
    for num_hidden in [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]:
        
        print("learning rate %g; hidden_num %d; dataset size %d" % (learning_rate, num_hidden, ds_values.shape[0]))
        total_accuracy = 0.0
    
        for _ in range(5):
            ds_1_data, ds_1_labels, ds_2_data, ds_2_labels = split(ds_values, ds_labels)
        
            mlp = MLP(learning_rate, 10001, num_hidden, ds_1_data, ds_1_labels, ds_2_data, ds_2_labels)
            accuracy = mlp.train(True, "Glass Identification")
            print("accuracy %g" % (accuracy))
            total_accuracy += accuracy
    
            mlp = MLP(learning_rate, 10001, num_hidden, ds_2_data, ds_2_labels, ds_1_data, ds_1_labels)
            accuracy = mlp.train(True, "Glass Identification")
            print("accuracy %g" % (accuracy))
            total_accuracy += accuracy

        total_accuracy /= 10
        print("average accuracy %g" % (total_accuracy))
        print()

print("The end.")
            