'''
Created on Oct 29, 2016

@author: anton
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_2d_dataset(x, y, figsize=(4.1, 4.1), savefig=None):
    if y.ndim == 2: y = y.argmax(axis=1)
    
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.scatter(x[:, 0], x[:, 1], c=y)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def plot_decision_regions(x, y, classifier, reject=None,
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
    Z = outputs
    if reject_output: Z = Z[:,:-1]
    Z = Z.argmax(axis=1)
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
    else: plt.close()

def plot_binary_roc_curve(fpr, tpr, roc_auc, savefig=None, show=True):
    colors = plt.cm.rainbow(np.linspace(0,1,4))  # @UndefinedVariable
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
    else: plt.close()

def plot_roc_curves(curves, savefig=None, show=True):
    colors = plt.cm.rainbow(np.linspace(0,1,curves.shape[0]))  # @UndefinedVariable
    label_auc = ' (AUC: {0:0.4f})'
    
    plt.figure(figsize=(4.1, 4.1))
    for curve, color in zip(curves, colors):
        fpr, tpr, _, auc, label = curve
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=label + label_auc.format(auc))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC curves for single threshold')
    plt.legend(loc="lower right")
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    else: plt.close()

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