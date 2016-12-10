'''
Created on Dec 7, 2016

@author: anton
'''

import numpy as np
from input_data import get_data
from data_utils import print_frequencies, calc_precision_recall
from data_utils import calc_roc
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
from graphics import draw_roc, draw_precision_recall
import itertools

def input_fn(x, y, n_classes):
    feature_cols = {str(k): tf.constant(x[:,k]) for k in range(0, n_classes)}
    labels = tf.constant(y)
    return feature_cols, labels

def main(unused_argv):
    x, y, n_classes = get_data(2, False, 0)
    print_frequencies(y)
    x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=.2, random_state=42)
    print_frequencies(y_trn)
    print_frequencies(y_tst)
    
    feature_columns = [tf.contrib.layers.real_valued_column(str(k)) for k in range(0, n_classes)]
    classifier = learn.DNNClassifier(feature_columns=feature_columns, 
                                     hidden_units=[10, 20, 10], 
                                     n_classes=n_classes)

    classifier.fit(input_fn=lambda: input_fn(x_trn, y_trn, n_classes), steps=20000)
    y = classifier.predict_proba(input_fn=lambda: input_fn(x_tst, y_tst, n_classes))
    outputs = list(itertools.islice(y, x_tst.shape[0]))
    
    predictions = [np.argmax(o) for o in outputs]
    score = metrics.accuracy_score(y_tst, predictions)
    print('Accuracy: {0:f}'.format(score))
    #
    fpr, tpr, roc_auc = calc_roc(y_tst, outputs)
    draw_roc(fpr, tpr, roc_auc)
    #
    precision, recall, average_precision = calc_precision_recall(y_tst, outputs)
    draw_precision_recall(precision, recall, average_precision)
    
if __name__ == '__main__':
    tf.app.run()