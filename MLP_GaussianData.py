'''
Created on Oct 29, 2016

@author: anton
'''

import numpy as np
from InputData import create_gaussian_data_2
from MLP import MLP
from Graphics import draw
from Graphics import draw_x_vs_y


"""
    MLP with softmax or sigmoid activation
    3 classes
    inside error only
    
    add_noise=None - no noise
    add_noise=1    - add noise to the training set
    add_noise=2    - add noise as a class
    
    add_noise=11   - add noise to the training set with sigmoid with outputs 0.1 and 0.9
"""
def test(num_hidden=3, num_steps=40001, size=400, activation_function='sigmoid', early_stopping=None, add_noise=None, graphics=True):
    trn_x=None
    trn_y=None
    vld_x=None 
    vld_y=None
    tst_x=None
    tst_y=None
    tst_outliers=None
    
    if add_noise is None:
        trn_x, trn_y, _            = create_gaussian_data_2(size, graphics=False)
        vld_x, vld_y, _            = create_gaussian_data_2(size, graphics=False)
        tst_x, tst_y, tst_outliers = create_gaussian_data_2(1000, graphics=graphics)
    elif add_noise==1:
        if activation_function=='softmax':
            trn_x, trn_y, _            = create_gaussian_data_2(size, add_noise=2,    noise_output=1.0/3.0, graphics=False)
            vld_x, vld_y, _            = create_gaussian_data_2(size, add_noise=None, graphics=False)
            tst_x, tst_y, tst_outliers = create_gaussian_data_2(1000, add_noise=1,    graphics=graphics)
        else:
            trn_x, trn_y, _            = create_gaussian_data_2(size, add_noise=2,    graphics=False)
            vld_x, vld_y, _            = create_gaussian_data_2(size, add_noise=None, graphics=False)
            tst_x, tst_y, tst_outliers = create_gaussian_data_2(1000, add_noise=1,    graphics=graphics)
    elif add_noise==2:
        trn_x, trn_y, _ = create_gaussian_data_2(size, add_noise=3, graphics=False)
        vld_x, vld_y, _ = create_gaussian_data_2(size, add_noise=3, graphics=False)
        tst_x, tst_y, _ = create_gaussian_data_2(1000, add_noise=3, graphics=graphics)
    elif add_noise==11:
        if activation_function=='softmax':
            assert False
        else:
            trn_x, trn_y, _            = create_gaussian_data_2(size, add_noise=2, max_output=0.9, min_output=0.1, noise_output=0.1, graphics=False)
            vld_x, vld_y, _            = create_gaussian_data_2(size, add_noise=None, graphics=False)
            tst_x, tst_y, tst_outliers = create_gaussian_data_2(1000, add_noise=1,    graphics=graphics)
    else:
        assert False

    mlp = MLP(0.1, trn_x.shape[1], num_hidden, trn_y.shape[1], activation_function=activation_function)

    trn_acc, vld_acc, loss = mlp.train(num_steps, trn_x, trn_y, vld_x, vld_y, early_stopping=early_stopping, logging=False)
    tst_acc = mlp.test(tst_x, tst_y, logging=False)
    str1 = ""
    
    if add_noise == 1:
        if early_stopping is not None:
            str1 = "-\\\\%g\\\\%g\\\\%g" % (vld_acc*100.0, tst_acc*100.0, loss)
            print("Training accuracy\\\\Validation accuracy\\\\Test accuracy\\\\Loss: " + str1)
        else:
            str1 = "-\\\\-\\\\%g\\\\%g" % (tst_acc*100.0, loss)
            print("Training accuracy\\\\Test accuracy\\\\Loss: " + str1)
    else:
        if early_stopping is not None:
            str1 = "%g\\\\%g\\\\%g\\\\%g" % (trn_acc*100.0, vld_acc*100.0, tst_acc*100.0, loss)
            print("Training accuracy\\\\Validation accuracy\\\\Test accuracy\\\\Loss: " + str1)
        else:
            str1 = "%g\\\\-\\\\%g\\\\%g" % (trn_acc*100.0, tst_acc*100.0, loss)
            print("Training accuracy\\\\Test accuracy\\\\Loss: " + str1)
    
    if graphics==True:
        for treshold in [0.2, 0.4, 0.6, 0.8]:
            c, e, rc, re = mlp.test_cer(tst_x, tst_y, outliers=tst_outliers, threshold=treshold, threshold_method=0, logging=False)
            draw(c, e, rc, re)
    
    r_0, c_0, re_0, rc_0 = mlp.test_rejection(tst_x, tst_y, outliers=tst_outliers, threshold_method=0)
    r_1, c_1, re_1, rc_1 = mlp.test_rejection(tst_x, tst_y, outliers=tst_outliers, threshold_method=1)
    r_2, c_2, re_2, rc_2 = mlp.test_rejection(tst_x, tst_y, outliers=tst_outliers, threshold_method=2)
    
    area0 = np.trapz(c_0, r_0)
    area1 = np.trapz(c_1, r_1)
    area2 = np.trapz(c_2, r_2)
    
    str2=""
    
    if add_noise == 1:
        str2 = "%g\\\\%g\\\\-" % (area0, area1)
    else:
        str2 = "%g\\\\%g\\\\%g" % (area0, area1, area2)
        
    print("areas under ROC: " + str2)
    
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
    
    return str1 + "\\\\" +str2
    
"""
test across hidden layer size
"""
def generate_table_hidden(activation_function='sigmoid', early_stopping=None, add_noise=None):
    results = []
    for num_hidden in [3, 4, 5, 6, 7]:
        for attempt in range(1,4):
            print("attempt: %d, hidden layer size: %d" % (attempt, num_hidden))
            result = test(num_hidden=num_hidden, activation_function=activation_function, early_stopping=early_stopping, add_noise=add_noise, graphics=False)
            results.append(result)
            print(result)
            
    print(results)
    
    with open("test.txt", "a+") as log_file:
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------------------", file=log_file)
        print("activation function: %s, early stopping: %s, outliers: %s" % (activation_function, early_stopping, add_noise), file=log_file)
        print("--------------------------------------------------------------------------------------------------------------------------------------", file=log_file)
        print("Test 1 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[0], results[3], results[6],  results[9], results[12]), file=log_file)
        print("\hline", file=log_file)
        print("Test 2 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[1], results[4], results[7],  results[10], results[13]), file=log_file)
        print("\hline", file=log_file)
        print("Test 3 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[2], results[5], results[8], results[11], results[14]), file=log_file)
    

def generate_table_steps(activation_function='sigmoid', add_noise=None, description=""):
    results = []
    for attempt in range(1,4):
        for num_steps in [1001, 2001, 4001, 8001, 16001, 32001, 40001]:
            print("attempt: %d, training steps: %d" % (attempt, num_steps))
            result = test(num_hidden=6, num_steps=num_steps, activation_function=activation_function, early_stopping=None, add_noise=add_noise, graphics=False)
            results.append(result)
            print(result)
            
    print(results)
            
    with open("test.txt", "a+") as log_file:
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------------------", file=log_file)
        print("activation function: %s, outliers: %s" % (activation_function, add_noise), file=log_file)
        print("--------------------------------------------------------------------------------------------------------------------------------------", file=log_file)
        print("Test 1 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[0], results[1],   results[2],   results[3],  results[4], results[5],  results[6]), file=log_file)
        print("\hline", file=log_file)
        print("Test 2 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[7], results[8],   results[9],  results[10], results[11], results[12], results[13]), file=log_file)
        print("\hline", file=log_file)
        print("Test 3 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[14], results[15], results[16], results[17], results[18], results[19], results[20]), file=log_file)
 
 
def generate_table_class_size(activation_function='sigmoid', early_stopping=None, add_noise=None, description=""):
    results = []
    for attempt in range(1,4):
        for class_size in [4, 16, 32, 64, 128, 256, 400]:
            print("attempt: %d, class_size: %d" % (attempt, class_size))
            result = test(num_hidden=6, size=class_size, activation_function=activation_function, early_stopping=early_stopping, add_noise=add_noise, graphics=False)
            results.append(result)
            print(result)
            
    print(results)
            
    with open("test.txt", "a+") as log_file:
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------------------------------------", file=log_file)
        print("activation function: %s, early stooping: %s, outliers: %s" % (activation_function, early_stopping, add_noise), file=log_file)
        print("--------------------------------------------------------------------------------------------------------------------------------------", file=log_file)
        print("Test 1 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[0], results[1],   results[2],   results[3],  results[4], results[5],  results[6]), file=log_file)
        print("\hline", file=log_file)
        print("Test 2 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[7], results[8],   results[9],  results[10], results[11], results[12], results[13]), file=log_file)
        print("\hline", file=log_file)
        print("Test 3 & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} & \shortstack{%s} \\\\" % (results[14], results[15], results[16], results[17], results[18], results[19], results[20]), file=log_file)         

######################################################################################
# tests across hidden layer size
######################################################################################

#generate_table_hidden(activation_function='sigmoid', early_stopping=None)
#generate_table_hidden(activation_function='softmax', early_stopping=None)
#generate_table_hidden(activation_function='sigmoid', early_stopping=50)
#generate_table_hidden(activation_function='softmax', early_stopping=50)

#generate_table_hidden(activation_function='sigmoid', early_stopping=None, add_noise=1)
generate_table_hidden(activation_function='sigmoid', early_stopping=None, add_noise=11)
#generate_table_hidden(activation_function='softmax', early_stopping=None, add_noise=1)
#generate_table_hidden(activation_function='sigmoid', early_stopping=50,   add_noise=1)
#generate_table_hidden(activation_function='sigmoid', early_stopping=50, add_noise=11)
#generate_table_hidden(activation_function='softmax', early_stopping=50,   add_noise=1)

#generate_table_hidden(activation_function='sigmoid', early_stopping=None, add_noise=2)
#generate_table_hidden(activation_function='softmax', early_stopping=None, add_noise=2)
#generate_table_hidden(activation_function='sigmoid', early_stopping=50,   add_noise=2)
#generate_table_hidden(activation_function='softmax', early_stopping=50,   add_noise=2)

######################################################################################
# tests across training steps
######################################################################################

#generate_table_steps(activation_function='sigmoid', add_noise=1)
#generate_table_steps(activation_function='softmax', add_noise=1)
#generate_table_steps(activation_function='sigmoid', add_noise=2)
#generate_table_steps(activation_function='softmax', add_noise=2)

######################################################################################
# tests across data size
######################################################################################

#generate_table_class_size(activation_function='sigmoid', early_stopping=None, add_noise=1)
#generate_table_class_size(activation_function='softmax', early_stopping=None, add_noise=1)
#generate_table_class_size(activation_function='sigmoid', early_stopping=50,   add_noise=1)
#generate_table_class_size(activation_function='softmax', early_stopping=50,   add_noise=1)

#generate_table_class_size(activation_function='sigmoid', early_stopping=None, add_noise=2)
#generate_table_class_size(activation_function='softmax', early_stopping=None, add_noise=2)
#generate_table_class_size(activation_function='sigmoid', early_stopping=50,   add_noise=2)
#generate_table_class_size(activation_function='softmax', early_stopping=50,   add_noise=2)

###############################