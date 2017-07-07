'''
Created on Jul 06, 2016

@author: anton
'''

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def remove_correct(cm, labels, error_threshold):
    n_classes = len(labels)
    delete_x = [i for i in range(n_classes)
                if cm[:,i].sum() - cm[i,i] <= error_threshold]
    delete_y = [i for i in range(n_classes)
                if cm[i,:].sum() - cm[i,i] <= error_threshold]
    
    labels_x = np.delete(labels, delete_x)
    labels_y = np.delete(labels, delete_y)
    
    cm = np.delete(cm, delete_x, axis=1)
    cm = np.delete(cm, delete_y, axis=0)
    
    return cm, labels_x, labels_y

def print_misslcassification_errors(cm, labels, limit=8):
    pairs = []
    values = []
    for r, c in itertools.combinations(enumerate(labels), 2):
        i1, l1 = r
        i2, l2 = c
        pairs.append([l1,l2])
        values.append(cm[i1,i2] + cm[i2,i1])
    values, pairs = zip(*sorted(zip(values, pairs), reverse=True))
    print(list(zip(pairs, values))[:limit])

def print_output_errors(cm, labels, limit=8):
    pairs = []
    values = []
    for i, l in enumerate(labels):
        pairs.append(l)
        values.append(cm[:,i].sum() - cm[i,i])
    values, pairs = zip(*sorted(zip(values, pairs), reverse=True))
    print(list(zip(pairs, values))[:limit])

def plot_confusion_matrix(y_true, y_pred, labels, savefig=None, show=True,
                          error_threshold=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    y_true : array, shape = [n_samples]
    Ground truth (correct) target values.
    
    y_pred : array, shape = [n_samples]
    Estimated targets as returned by a classifier.
    
    labels : array, shape = [n_classes]
    Names of classes for columns and rows in the final matrix
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection
    /plot_confusion_matrix.html
    #sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    
    x_labels = np.copy(labels)
    y_labels = np.copy(labels)
    
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    print(cm)
    print_misslcassification_errors(cm, labels)
    print_output_errors(cm, labels)
    
    if error_threshold is not None:
        cm, x_labels, y_labels = remove_correct(cm, labels, error_threshold)
    
    fig = plt.figure()
    title = 'Confusion matrix with error threshold ' + str(error_threshold)
    fig.canvas.set_window_title(title)
    plt.imshow(cm, interpolation='nearest',
               cmap=plt.cm.Blues)  # @UndefinedVariable
    #plt.colorbar()
    
    x_tick_marks = np.arange(len(x_labels))
    y_tick_marks = np.arange(len(y_labels))
    plt.xticks(x_tick_marks, x_labels, rotation=45)
    plt.yticks(y_tick_marks, y_labels)
    
    #if normalize:
    #    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print('Normalized confusion matrix')
    #else:
    #    print('Confusion matrix, without normalization')
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savefig is not None: plt.savefig(savefig)
    if show:    plt.show()
    else:       plt.close()