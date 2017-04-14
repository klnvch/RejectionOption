'''
Created on Nov 12, 2016

@author: anton

test thresholding
'''

from graphics import plot_binary_roc_curve, plot_confusion_matrix, plot_multiclass_roc_curve
from data_utils import calc_roc_binary, calc_roc_multiclass
from MLP import MLP
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from DataSet import DataSet

def test(ds, activation_function='softmax', optimizer='adagrad', learning_rate=0.1, steps=100001, batch_size=256, hidden_layer=[2], regularization_penalty=0.0, graphics=False):
    print('learning rate {:g}; hidden_num {:s}'.format(learning_rate, str(hidden_layer)))
        
    mlp = MLP(learning_rate, [ds.num_features]+hidden_layer+[ds.num_classes], activation_function, optimizer, regularization_penalty)
    result = mlp.train(steps, ds.trn.x, ds.trn.y, ds.vld.x, ds.vld.y, batch_size, logging=True)
    
    print(result)
        
    outputs = mlp.predict_proba(ds.tst.x, result[0])
    predictions = [np.argmax(o) for o in outputs]
    y_true = [np.argmax(y) for y in ds.tst.y]
    accuracy_score = metrics.accuracy_score(y_true, predictions)
    print('Accuracy: {0:f}'.format(accuracy_score))
    print(metrics.classification_report(y_true, predictions))
    
    fpr_binary, tpr_binary, roc_auc_binary = calc_roc_binary(ds.tst.y, outputs)
    fpr_multiclass, tpr_multiclass, roc_auc_multiclass = calc_roc_multiclass(ds.tst.y, outputs, ds.class_names)
    
    if graphics:
        # plot confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_true, predictions)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=ds.class_names)
        plt.show()
        
        #plot ROC curves for multiple output thresholds
        # Compute ROC curve and ROC area for each class
        plt.figure()
        plot_multiclass_roc_curve(fpr_multiclass, tpr_multiclass, roc_auc_multiclass, ds.class_names)
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
    ds = DataSet(1)
    #
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [28], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [32], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [36], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 256, [40], graphics=False)
    #
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [28], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [32], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [36], graphics=False)
    #test(ds, 'sigmoid', 'adagrad', 0.01, 20000, 512, [40], graphics=False)
    #
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [28], graphics=False)
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [32], graphics=False)
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [36], graphics=False)
    #test(ds, 'softmax', 'adagrad', 0.01, 20000, 256, [40], graphics=False)
    #
    #test(ds, 'sigmoid', 'gradient', 0.01, 20000, 256, [20,20], graphics=False)
    #test(ds, 'sigmoid',     'Adam', 0.01, 20000, 256, [20,20], graphics=False)
    #test(ds, 'sigmoid',  'adagrad', 0.01, 20000, 256, [20,20], graphics=False)
    #
    #test(ds, 'sigmoid',     'Adam', 0.01, 2000, 128, [20,20], graphics=False)
    #test(ds, 'sigmoid',     'Adam', 0.01, 2000, 512, [20,20], graphics=False)
    #
    #test(ds, 'sigmoid',     'Adam', 0.01, 2000, 128, [20,20], 0.5, graphics=False)
    #test(ds, 'sigmoid',     'Adam', 0.01, 2000, 128, [20,20], 1.0, graphics=False)
    
    # one month cancel_dump_traceback_later
    
    # compare different optimizers, just for fun, result in optimizers.txt
    """
    table = ''
    for optimizer in ['GradientDescent', 'Adadelta', 'Adagrad', 'Momentum', 'Adam', 'Ftrl', 'RMSProp']:
        for learning_rate in [0.1, 0.01, 0.001]:
            for _ in range(2):
                values = test(ds, 'sigmoid', optimizer, learning_rate, 10000, 256, [32], graphics=False)
                line = '{:.4f}&{:.4f}&{:.4f}&{:.6f}&{:.6f}&{:.6f}&{:.6f}&{:.6f}'.format(values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9])
                table += line + ' | ' + optimizer + ' | ' + str(learning_rate) + '\n'
    print(table)            
    """
    # test number of nodes in the hiden layer
    """
    table = ''
    #for hidden in [[12],[16],[20],[24],[28],[32],[36],[40],[44],[48],[52],[56],[60]]:
    #for hidden in [[8,8],[16,16],[20,20],[24,24],[28,28],[32,32],[12,28],[28,12],[8,24],[24,8]]:
    #for hidden in [[8,8,8], [12,12,12], [16,16, 16], [20, 20, 20], [12,20,28], [28,20,12]]:
    #for hidden in [[12],[16],[20],[24],[28],[32],[36],[40],[44],[48],[52],[56],[60],[8,8],[16,16],[20,20],[24,24],[28,28],[32,32],[12,28],[28,12],[8,24],[24,8],[8,8,8], [12,12,12], [16,16, 16], [20, 20, 20], [12,20,28], [28,20,12]]:
    for hidden in [[1],[2],[4],[6],[8],[10],[12],[14],[16],[18],[20],[24],[30],[2,2],[4,4],[8,8],[10,10],[12,12],[16,16],[12,28],[28,12],[4,16],[16,4],[4,4,4], [6,6,6], [8,8,8], [10, 10, 10], [4,8,12], [12,8,4]]:
        for _ in range(1):
            values = test(ds, 'sigmoid', 'Adam', 0.01, 10000, 256, hidden, graphics=False)
            line = '{:.4f}&{:.4f}&{:.4f}&{:.6f}&{:.6f}&{:.6f}&{:.6f}&{:.6f}'.format(values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9])
            table += str(hidden) + '&' + line + '\\\\\n'
            table += '\\hline\n'
    print(table)
    """
    test(ds, 'sigmoid', 'Adam', 0.01, 10000, 256, [16], graphics=True)
    