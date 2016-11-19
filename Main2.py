'''
Created on Nov 12, 2016

@author: anton
'''
from InputData import get_data
from DataUtils import split_in_tvt
from DataUtils import split_in_tt
from DataUtils import remove_class
from DataUtils import add_noise_as_no_class
from DataUtils import add_noise_as_a_class
from Graphics import draw_x_vs_y
from MLP import MLP
import numpy as np

def test(ds, num_steps=100001, num_hidden=2, activation_function='softmax', early_stopping=50, graphics=True):
    print("learning rate %g; hidden_num %d" % (0.1, num_hidden))
    
    trn_x=None
    trn_y=None
    vld_x=None 
    vld_y=None
    tst_x=None
    tst_y=None
    outliers=None
    
    if early_stopping is None:
        trn_x, trn_y, tst_x, tst_y = ds
        vld_x = tst_x
        vld_y = tst_y
    else:
        trn_x, trn_y, vld_x, vld_y, tst_x, tst_y = ds
        
    mlp = MLP(0.1, trn_x.shape[1], num_hidden, trn_y.shape[1], activation_function)
    result = mlp.train(num_steps, trn_x, trn_y, vld_x, vld_y, early_stopping, False)
        
    print(result[0])
    print(result[1])
    print(result[2])
    print(result[3])
        
    tst_acc_0 = mlp.test(tst_x, tst_y, result[0][0], logging=True)
    tst_acc_1 = mlp.test(tst_x, tst_y, result[1][0], logging=True)
    tst_acc_2 = mlp.test(tst_x, tst_y, result[2][0], logging=True)
    tst_acc_3 = mlp.test(tst_x, tst_y, result[3][0], logging=True)
        
    print([tst_acc_0, tst_acc_1, tst_acc_2, tst_acc_3])
    
    r_0, c_0, re_0, rc_0 = mlp.test_rejection(tst_x, tst_y, outliers=outliers, threshold_method=0, rejection_rate_limit=50, filename=result[1][0])
    r_1, c_1, re_1, rc_1 = mlp.test_rejection(tst_x, tst_y, outliers=outliers, threshold_method=1, rejection_rate_limit=50, filename=result[2][0])
    r_2, c_2, re_2, rc_2 = mlp.test_rejection(tst_x, tst_y, outliers=outliers, threshold_method=2, rejection_rate_limit=50, filename=result[3][0])
    
    area0 = np.trapz(c_0, r_0)
    area1 = np.trapz(c_1, r_1)
    area2 = np.trapz(c_2, r_2)
    
    str2 = '{:f}\\\\{:f}\\\\{:f}'.format(area0, area1, area2)
    
    if graphics==True:
        draw_x_vs_y([r_0, r_1, r_2], [c_0, c_1, c_2], 
                'Rejecting rate, %', 'Accuracy after rejection, %', 
                legend_location=4,
                labels=['output', 'difference', 'ratio'], 
                colors=['b', 'g', 'r'])
        draw_x_vs_y([re_0, re_1, re_2], [rc_0, rc_1, rc_2], 
                'Rejected errors, %', 'Rejected correct samples, %', 
                legend_location=2,
                labels=['output', 'difference', 'ratio'], 
                colors=['b', 'g', 'r'])

    return str2

def generate_table_hidden_size(num_steps=100001, activation_function='softmax', early_stopping=None):
    ds_x, ds_y = get_data(3)
    
    for attempt in [1]:
        ds = None
        if early_stopping is None:
            ds = split_in_tt(ds_x, ds_y)
        else:
            ds = split_in_tvt(ds_x, ds_y)
            
        results=[]
            
        for num_hidden in [3, 4, 6, 8, 9, 10, 12]:
            print('attempt: {:d}, hidden layer size: {:d}'.format(attempt, num_hidden))
            result = test(ds, num_steps, num_hidden, activation_function, early_stopping, False)
            results.append(result)
            print(result)
            
        with open('test.txt', 'a+') as log_file:
            print('-------------------------------------------------------------------------------------------', file=log_file)
            print('activation function: {:s}, early_stopping: {:d}'.format(activation_function, early_stopping), file=log_file)
            print('-------------------------------------------------------------------------------------------', file=log_file)
            print(results, file=log_file)
            print('-------------------------------------------------------------------------------------------', file=log_file)

            

#generate_table_hidden_size(100001, 'sigmoid', 100)
generate_table_hidden_size(6001, 'softmax', 100)