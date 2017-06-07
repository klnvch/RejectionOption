'''
Created on Oct 29, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.colors as colors
import matplotlib.cm as cmx

def plot_2d_dataset(x, y, figsize=(4.1, 4.1), savefig=None):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.scatter(x[:, 0], x[:, 1], c=y)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def plot_decision_regions(x, y, classifier, reject=None,
                          figsize=(4.1, 4.1), savefig=None, show=True,
                          step_size=0.02):
    # create a mesh to plot in
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    outputs = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # imshow_handle = plt.imshow(Z, interpolation='nearest', 
    #           extent=(x_min, x_max, y_min, y_max), aspect='auto', 
    #           origin="lower", cmap=plt.cm.PuOr_r)  # @UndefinedVariable
    Z = outputs.argmax(axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)  # @UndefinedVariable
    if reject is not None:
        scores = reject(outputs)
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
    else: plt.clf()

def plot_binary_roc_curve(fpr, tpr, roc_auc, savefig=None, show=True):
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    labels = ['Single output threshold (AUC: {0:0.4f})',
              'Single differential threshold (AUC: {0:0.4f})',
              'Single ratio threshold (AUC: {0:0.4f})',
              'Multiple output thresholds (AUC: {0:0.4f})']
    plt.figure()
    for i, color, label in zip([0, 1, 2, 'm'], colors, labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=label.format(roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curves for single threshold')
    plt.legend(loc="lower right")
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.clf()
    
def draw_precision_recall(precision, recall, average_precision, filename=None):
    colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip([0, 1, 2], colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label='Precision-recall curve of method {0} (area = {1:0.4f})'.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves for single treshold')
    plt.legend(loc="lower right")
    #
    if filename is not None:
        plt.savefig(filename)

def plot_confusion_matrix(y_true, y_pred, labels, savefig=None, show=True):
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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # @UndefinedVariable
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    #if normalize:
    #    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print('Normalized confusion matrix')
    #else:
    #    print('Confusion matrix, without normalization')
    
    np.set_printoptions(precision=2)
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.clf()

def plot_multiclass_roc_curve(fpr, tpr, roc_auc, labels,
                              savefig=None, show=True):
    """
    Plot ROC curves for the multiclass problem
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection
    /plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    """
    n_classes = len(labels)
    # Plot all ROC curves
    lw = 2
    colors = plt.cm.rainbow(np.linspace(0,1,n_classes))  # @UndefinedVariable
    plt.figure()
    for i, color in zip(range(n_classes), colors):
        label = 'Class ''{0}'' (AUC: {1:0.4f})'.format(labels[i], roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=label)
    
    label_micro = 'Micro-average (AUC: {0:0.4f})'.format(roc_auc["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=label_micro, color='deeppink',
             linestyle=':', linewidth=4)
    label_macro = 'Macro-average (AUC: {0:0.4f})'.format(roc_auc["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=label_macro, color='navy',
             linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curves for multiple thresholds')
    plt.legend(loc="lower right")
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.clf()

def plot_multiclass_precision_recall_curve(y_test, y_score, class_names):
    """
    Plot Precision-Recall curves for the multiclass problem
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    """
    n_classes = len(class_names)
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")

    # Plot Precision-Recall curve for each class
    lw = 2
    colors = get_cmap(n_classes)
    
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], color=colors(i), lw=lw,
             label='Precision-recall curve of class ''{0}'' (area = {1:0.4f})'
                   ''.format(class_names[i], average_precision[i]))

    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
        label='micro-average Precision-recall curve (area = {0:0.4f})'
               ''.format(average_precision["micro"]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves for multiple thresholds')
    plt.legend(loc="lower right")
    



def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.
    
    Copied from http://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    '''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color
    
