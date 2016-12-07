'''
Created on Dec 7, 2016

@author: anton
'''

import numpy as np
from InputData import get_data
from DataUtils import rejection_score
from DataUtils import print_frequencies
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
from itertools import cycle


def main(unused_argv):
    x, y, n_classes = get_data(3, False, 3)
    print_frequencies(y)
    x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=.2, random_state=42)
    print_frequencies(y_trn)
    print_frequencies(y_tst)
    
    feature_columns = learn.infer_real_valued_columns_from_input(x_trn)
    classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10], n_classes=n_classes)

    classifier.fit(x_trn, y_trn, steps=20000)
    outputs = list(classifier.predict_proba(x_tst, as_iterable=True))
    predictions = [np.argmax(o) for o in outputs]
    score = metrics.accuracy_score(y_tst, predictions)
    print('Accuracy: {0:f}'.format(score))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_true = y_tst - predictions
    
    for i in [0,1,2]:
        y_score = rejection_score(outputs, i)
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score, pos_label=0)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    
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
    
    
if __name__ == '__main__':
    tf.app.run()