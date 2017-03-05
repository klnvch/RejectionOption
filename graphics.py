'''
Created on Oct 29, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def draw(c, e, rc, re, axis=1):
    
    if c.shape[0] == 0: c = np.array([[None, None]])
    if e.shape[0] == 0: e = np.array([[None, None]])
    if rc.shape[0] == 0: rc = np.array([[None, None]])
    if re.shape[0] == 0: re = np.array([[None, None]])
    
    lbl_c,  = plt.plot(*zip(*c),  marker='s', color='green', markersize='5', ls='', label='Correct')
    lbl_e,  = plt.plot(*zip(*e),  marker='p', color='red',   markersize='5', ls='', label='Errors')
    lbl_rc, = plt.plot(*zip(*rc), marker='o', color='blue',  markersize='5', ls='', label='Rejected correct')
    lbl_re, = plt.plot(*zip(*re), marker='o', color='cyan',  markersize='5', ls='', label='Rejected errors')
    
    plt.legend(handles=[lbl_c, lbl_e, lbl_rc, lbl_re], numpoints=1, loc=2)
    plt.axis([-axis, axis, -axis, axis])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()


def draw_x_vs_y(xs, ys, xlabel=None, ylabel=None, labels=None, colors=None, legend_location=2):
    handles = []
    
    for x, y, label, color in zip(xs, ys, labels, colors):
        lbl, = plt.plot(x, y, color=color, linewidth=2.0, label=label)
        handles.append(lbl)
    
    plt.legend(handles=handles, numpoints=1, loc=legend_location)
    if xlabel is not None and ylabel is not None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.show()
    
def plot_binary_roc_curve(fpr, tpr, roc_auc, filename=None):
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip([0,1,2], colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of method {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curves for single threshold')
    plt.legend(loc="lower right")
    #
    if filename is not None:
        plt.savefig(filename)
    
    
def draw_precision_recall(precision, recall, average_precision, filename=None):
    colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip([0,1,2], colors):
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




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):  # @UndefinedVariable
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    cm - two-dimensional ndarray of integers - confusion matrix
    classes - one-dimensional ndarray of strings - class names
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def plot_multiclass_roc_curve(fpr, tpr, roc_auc, class_names):
    """
    Plot ROC curves for the multiclass problem
    
    Copied from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    
    y_test - two-dimensional ndarray of size n_classes x number of samples
    y_score - two-dimensional ndarray of size n_classes x number of samples
    """
    n_classes = len(class_names)

    # Plot all ROC curves
    lw = 2
    colors = get_cmap(n_classes)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=lw,
             label='ROC curve of class ''{0}'' (area = {1:0.4f})'
             ''.format(class_names[i], roc_auc[i]))
        
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curves for multiple thresholds')
    plt.legend(loc="lower right")




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
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color



def plot_pca_vs_lda(X, y, target_names):
    """
    Copied from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
    """
    n_classes = len(target_names)
    
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = get_cmap(n_classes)
    lw = 2

    for i in range(n_classes):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=colors(i), alpha=.8, lw=lw, s=4,
                label=target_names[i])
    plt.legend(loc='best', shadow=False, scatterpoints=1, markerscale=4)
    plt.title('PCA')

    plt.figure()
    for i in range(n_classes):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=colors(i), s=4,
                label=target_names[i])
    plt.legend(loc='best', shadow=False, scatterpoints=1, markerscale=4)
    plt.title('LDA')

    plt.show()
    
    
    
    
if __name__ == '__main__':
    # draw ROC space
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # roc discrete rejections
    #plt.plot([.1, .4, .7, .0, .6], [.6, .8, .7, 1., .2], 'ro')
    #plt.annotate('A', xy=(.12, .55))
    #plt.annotate('B', xy=(.42, .75))
    #plt.annotate('C', xy=(.72, .65))
    #plt.annotate('D', xy=(.02, .95))
    #plt.annotate('E', xy=(.62, .15))
    
    # three roc curves
    plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.96, 0.97, 0.98, 0.99, 1], 'r')
    plt.annotate('A', xy=(.05, .7))
    plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.93, 0.95, 0.97, 1], 'b')
    plt.annotate('B', xy=(.25, .65))
    plt.plot([0, 0.4, 1], [0, 0.8, 1], 'g')
    plt.annotate('C', xy=(.3, .55))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()