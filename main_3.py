'''
Created on Mar 5, 2017

@author: anton

test training with low output
'''
from input_data import get_data
from graphics import plot_binary_roc_curve, plot_confusion_matrix, plot_multiclass_roc_curve
from data_utils import calc_roc_binary, calc_roc_multiclass, split_dataset,\
    remove_class, add_noise_as_no_class
from MLP import MLP
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt

def test(ds, activation_function='softmax', optimizer='adagrad', learning_rate=0.1, steps=100001, batch_size=256, hidden_layer=[2], regularization_penalty=0.0, graphics=False):
    print('learning rate {:g}; hidden_num {:s}'.format(learning_rate, str(hidden_layer)))
        
    trn_x, trn_y, vld_x, vld_y, tst_x, tst_y, class_names, outliers = ds
        
    mlp = MLP(learning_rate, [trn_x.shape[1]]+hidden_layer+[trn_y.shape[1]], activation_function, optimizer, regularization_penalty)
    result = mlp.train(steps, trn_x, trn_y, vld_x, vld_y, batch_size, logging=True)
    
    print(result)
    
    # calculate accuracy without outliers
    outputs = mlp.predict_proba(tst_x, result[0])
    predictions = [np.argmax(o) for o in outputs]
    y_true = [np.argmax(y) for y in tst_y]
    accuracy_score = metrics.accuracy_score(y_true, predictions)
    print('Accuracy: {0:f}'.format(accuracy_score))
    print(metrics.classification_report(y_true, predictions))
    
    # build ROC curves with outliers
    outliers_outputs = mlp.predict_proba(outliers, result[0])
    fpr_binary, tpr_binary, roc_auc_binary = calc_roc_binary(tst_y, outputs, outliers_outputs)
    fpr_multiclass, tpr_multiclass, roc_auc_multiclass = calc_roc_multiclass(tst_y, outputs, class_names)
    
    if graphics:
        # plot confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_true, predictions)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names)
        plt.show()
        
        #plot ROC curves for multiple output thresholds
        # Compute ROC curve and ROC area for each class
        plt.figure()
        plot_multiclass_roc_curve(fpr_multiclass, tpr_multiclass, roc_auc_multiclass, class_names)
        plt.show()
        
        # plot ROC curve for different rejection methods
        plt.figure()
        plot_binary_roc_curve(fpr_binary, tpr_binary, roc_auc_binary, 'tests/roc_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        plt.show()
        
        #plot Precision-Recall curves for multiple output threshols
        #plt.figure()
        #plot_multiclass_precision_recall_curve(tst_y, outputs, class_names)
        #plt.show()
        
        # plot precision-recall curve
        #precision, recall, average_precision = calc_precision_recall(y_true, outputs)
        #plt.figure()
        #draw_precision_recall(precision, recall, average_precision, 'tests/prr_{:s}_{:d}.png'.format(str(hidden_layer), int(time.time())))
        #plt.show()
        
    return [result[1], result[2], result[3], result[4], accuracy_score, roc_auc_binary[0], roc_auc_binary[1], roc_auc_binary[2], roc_auc_multiclass["micro"], roc_auc_multiclass["macro"]]


if __name__ == '__main__':
    ds_x, ds_y, class_names = get_data(1, binarize=True, preprocess=1)
    ds_x, ds_y, class_names, outliers = remove_class(ds_x, ds_y, class_names, [0])
    trn_x, trn_y, vld_x, vld_y, tst_x, tst_y = split_dataset(ds_x, ds_y, 0.6, 0.5, random_state=42)
    trn_x, trn_y = add_noise_as_no_class(trn_x, trn_y, 12000, 0)
    ds = [trn_x, trn_y, vld_x, vld_y, tst_x, tst_y, class_names, outliers]
    
    # test number of nodes in the hiden layer
    """
    table = ''
    #for hidden in [[12],[16],[20],[24],[28],[32],[36],[40],[44],[48],[52],[56],[60]]:
    #for hidden in [[8,8],[16,16],[20,20],[24,24],[28,28],[32,32],[12,28],[28,12],[8,24],[24,8]]:
    #for hidden in [[8,8,8], [12,12,12], [16,16, 16], [20, 20, 20], [12,20,28], [28,20,12]]:
    for hidden in [[12],[16],[20],[24],[28],[32],[36],[40],[44],[48],[52],[56],[60],[8,8],[16,16],[20,20],[24,24],[28,28],[32,32],[12,28],[28,12],[8,24],[24,8],[8,8,8], [12,12,12], [16,16, 16], [20, 20, 20], [12,20,28], [28,20,12]]:
        for _ in range(1):
            values = test(ds, 'sigmoid', 'Adam', 0.01, 10000, 256, hidden, graphics=False)
            line = '{:.4f}&{:.4f}&{:.4f}&{:.6f}&{:.6f}&{:.6f}&{:.6f}&{:.6f}'.format(values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9])
            table += str(hidden) + '&' + line + '\\\\\n'
            table += '\\hline\n'
    print(table)
    """
    
    test(ds, 'sigmoid', 'Adam', 0.01, 10000, 256, [32], graphics=True)
    