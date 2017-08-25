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
                if cm[i,i] / cm[:,i].sum() >= error_threshold]
    delete_y = [i for i in range(n_classes)
                if cm[i,i] / cm[i,:].sum() >= error_threshold]
    
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

def plot_confusion_matrix(y_true, y_pred, y_outl, labels, savefig=None, show=True,
                          error_threshold=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    y_true : array, shape = [n_samples]
    Ground truth (correct) target values.
    
    y_pred : array, shape = [n_samples]
    Estimated targets as returned by a classifier.
    
    y_outl : array, shape = [n_outliers]
    Estimated targets of outliers as returned by a classifier.
    
    labels : array, shape = [n_classes]
    Names of classes for columns and rows in the final matrix
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection
    /plot_confusion_matrix.html
    #sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    
    # Add outliers class
    if y_outl is not None:
        y_outl = y_outl.argmax(axis=1)
        n_classes = len(labels)
        y_true = np.concatenate((y_true, np.full(y_outl.shape, n_classes,
                                                 dtype=int)))
        y_pred = np.concatenate((y_pred, y_outl))
        labels = np.concatenate((labels, ['Outliers']))
    
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

def plot_multiclass_curve(fpr, tpr, roc_auc, labels,
                          savefig=None, show=True, curve_func = 'roc'):
    """
    Plot ROC curves for the multiclass problem
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection
    /plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    """
    n_classes = len(labels)
    # Plot all ROC curves
    lw = 2
    colors = plt.cm.rainbow(np.linspace(0,1,n_classes))  # @UndefinedVariable
    fig = plt.figure()
    for i, color in zip(range(n_classes), colors):
        if roc_auc[i] == 0: continue
        label = 'Class ''{0}'' (AUC: {1:0.4f})'.format(labels[i], roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=label)
    
    if 'micro' in roc_auc and 'macro' in roc_auc:
        label_micro = 'Micro-average (AUC: {0:0.4f})'.format(roc_auc["micro"])
        plt.plot(fpr["micro"], tpr["micro"], label=label_micro,
                 color='deeppink', linestyle=':', linewidth=4)
        label_macro = 'Macro-average (AUC: {0:0.4f})'.format(roc_auc["macro"])
        plt.plot(fpr["macro"], tpr["macro"], label=label_macro,
                 color='navy', linestyle=':', linewidth=4)
    
    if curve_func == 'roc':
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    if curve_func == 'precision_recall':
        fig.canvas.set_window_title('Multiclass_Precision_Recall_Curve')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
    elif curve_func == 'roc':
        fig.canvas.set_window_title('Multiclass_ROC_curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    else: raise ValueError('wrong curve function')
    #plt.title('ROC curves for multiple thresholds')
    plt.legend(loc="lower right")
    
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.close()

def plot_curves(curves, savefig=None, show=True, curve_func = 'roc'):
    colors = plt.cm.rainbow(np.linspace(0,1,curves.shape[0]))  # @UndefinedVariable
    label_auc = ' (AUC: {0:0.4f})'
    
    fig = plt.figure(figsize=(4.1, 4.1))
    for curve, color in zip(curves, colors):
        fpr, tpr, _, auc, label = curve
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=label + label_auc.format(auc))
    
    if curve_func == 'roc':
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    if curve_func == 'precision_recall':
        fig.canvas.set_window_title('Single_Precision_Recall_Curve')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
    elif curve_func == 'roc':
        fig.canvas.set_window_title('Single_ROC_Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    #plt.title('ROC curves for single threshold')
    plt.legend(loc="lower right")
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.close()

def plot_decision_regions(x, y, clf, threshold_func=None,
                          figsize=(4.1, 4.1), savefig=None, show=True,
                          step_size=0.02, reject_output=False):
    """ Plot decision regions
    
    Args:
      x:
      y:
      classifier:
      figsize:
      savefig:
      show:
      step_szie:
      reject_otput: indicates if the last class ignored as the region
    """
    # delete rejection class, that is the last column in y
    if reject_output:
        x = [a for a,b in zip(x,y) if b.argmax()!=(len(b)-1)]
        x = np.array(x)
        y = [b for b in y if b.argmax()!=(len(b)-1)]
        y = np.array(y)
        y = y[:,:-1]
        
    # create a mesh to plot in
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    outputs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # imshow_handle = plt.imshow(Z, interpolation='nearest', 
    #           extent=(x_min, x_max, y_min, y_max), aspect='auto', 
    #           origin="lower", cmap=plt.cm.PuOr_r)  # @UndefinedVariable
    if reject_output: outputs = outputs[:,:-1]
    Z = outputs
    Z = Z.argmax(axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)  # @UndefinedVariable
    if threshold_func is not None:
        scores = threshold_func(outputs)
        scores = scores.reshape(xx.shape)
        cnt = plt.contourf(xx, yy, scores, cmap=plt.cm.Greys, alpha=.4)  # @UndefinedVariable
        plt.clabel(cnt, fmt='%2.1f', inline=False, colors='red', fontsize=14)
    plt.scatter(x[:, 0], x[:, 1], c=y.argmax(axis=1), s=6)
    plt.contour(xx, yy, Z, colors='white')
    plt.axis('off')

    # Plot also the training points
    # color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
    # colors = [color_map[_y] for _y in y]
    # plt.colorbar(imshow_handle, orientation='horizontal')
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.close()