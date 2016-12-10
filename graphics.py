'''
Created on Oct 29, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

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
    
def draw_roc(fpr, tpr, roc_auc):
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip([0,1,2], colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of method {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    
def draw_precision_recall(precision, recall, average_precision):
    plt.figure()
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip([0,1,2], colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label='Precision-recall curve of class {0} (area = {1:0.4f})'.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()