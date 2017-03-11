'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import heapq
from collections import Counter
from sklearn import metrics
from scipy import interp
from sklearn.model_selection import train_test_split

def count_distribution(y):
    d = [0] * y.shape[1]
    
    for i in y:
        d[i.argmax()] += 1
        
    #d = np.asarray(d) / ds_y.shape[0]
    print(d)
    return d

def print_frequencies(x):
    print(Counter(x))




def split_dataset(ds_x, ds_y, split1=0.6, split2=0.5, random_state=None):
    """
    Splits dataset into train, validation and test sets
    Args:
        ds_x: features
        ds_y: outputs
        split1: train set size from the entire set
        split2: validation set size from the rest set
    Returns:
        train x and y, validation x and y, test x an y
    """
    split_1 = train_test_split(ds_x, ds_y, test_size=split1, random_state=random_state)
    split_2 = train_test_split(split_1[0], split_1[2], test_size=split2, random_state=random_state)
    return split_1[1], split_1[3], split_2[0], split_2[2], split_2[1], split_2[3]
    



def remove_class(ds_x, ds_y, classes_names, i):
    """
    Removes classes from list i in dataset ds_x, ds_y
    Args:
        ds_x: features
        ds_y: binarized outputs
        classes_names: names of classes
        i: list of classes to be removed
    Returns:
        new dataset, ds_x,ds_y and outliers
    """
    new_ds_x = []
    new_ds_y = []
    new_classes_names = np.delete(classes_names, i)
    outliers = []
    for x, y in zip(ds_x, ds_y):
        if y.argmax() in i:
            outliers.append(x)
        else:
            new_ds_x.append(x)
            new_ds_y.append(np.delete(y, i))
    return np.array(new_ds_x), np.array(new_ds_y), new_classes_names, np.array(outliers)




def add_noise_as_no_class(ds_x, ds_y, noise_size=None, noise_output=None):
    """
    Adds noise to a dataset
    Args:
        ds_x: features
        ds_y: binarized outputs
        noise_size: numner of noise patterns, default is side of the dataset
        noise_output: noise output, defult is 1./number of classes
    Returns:
        new dataset, ds_x,ds_y and outliers
    """
    assert ds_x.shape[0] == ds_y.shape[0]
    
    if noise_size is None:
        noise_size = ds_y.shape[0]
        
    if noise_output is None:
        noise_output = 1.0/ds_y.shape[1]
    
    noise = np.random.uniform(ds_x.min(), ds_x.max(), [noise_size, ds_x.shape[1]])
    
    new_ds_x = np.concatenate([ds_x, noise])
    new_ds_y = np.concatenate([ds_y, np.array([[noise_output]*ds_y.shape[1]] * noise_size)])
    
    assert new_ds_x.shape[0] == new_ds_y.shape[0]
    return new_ds_x, new_ds_y




def add_noise_as_a_class(ds_x, ds_y, noise_size=None):
    assert ds_x.shape[0] == ds_y.shape[0]
    
    if noise_size is None:
        noise_size = ds_y.shape[0]
    
    noise = np.random.uniform(0.0, 1.0, [noise_size, ds_x.shape[1]])
    
    ds_y = np.append(ds_y, np.array([[0]] * ds_y.shape[0]), axis=1)
    
    new_ds_x = np.concatenate([ds_x, noise])
    new_ds_y = np.concatenate([ds_y, np.array([[0]*ds_y.shape[1] + [1]] * noise_size)])
    
    assert new_ds_x.shape[0] == new_ds_y.shape[0]
    return new_ds_x, new_ds_y

def rejection_score(outputs, rejection_method):
    if rejection_method == 0:
        return np.max(outputs, axis=1)
    elif rejection_method == 1:
        result = []
        for o in outputs:
            x = heapq.nlargest(2, o)
            result.append(x[0] - x[1])
        return result
    elif rejection_method == 2:
        result = []
        for o in outputs:
            x = heapq.nlargest(2, o)
            result.append(1.0 - x[1] / x[0])
        return result
    else:
        assert False




def calc_roc_binary(y_test, outputs, outliers_outputs=None):
    """
    Calcs binary ROC curve or for single threshold
    Args:
        y_test: desired output
        outputs: real output
        outliers_outputs: output for outliers
    Returns:
        FPR, TPR, area under the ROC curve
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y = [np.argmax(y) for y in y_test]
    predictions = [np.argmax(o) for o in outputs]
    y_true = [a==b for a,b in zip(np.array(y), np.array(predictions))]
    
    for i in [0,1,2]:
        if outliers_outputs is None:
            y_score = rejection_score(outputs, i)
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_score)
        else:
            y_score = rejection_score(np.concatenate((outputs, outliers_outputs),axis=0), i)
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true + [False] * outliers_outputs.shape[0], y_score)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    return fpr, tpr, roc_auc




def calc_roc_multiclass(y_test, outputs, class_names, outliers_outputs=None):
    """
    Calcs binary ROC curve or for multiple threshold
    Args:
        y_test: desired output
        outputs: real output
        class_names: names of classes
        outliers_outputs: output for outliers
    Returns:
        FPR, TPR, area under the ROC curve
    """
    n_classes = len(class_names)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y = [np.argmax(y) for y in y_test]
    predictions = [np.argmax(o) for o in outputs]
    y_true = [a==b for a,b in zip(np.array(y), np.array(predictions))]
    y_score = rejection_score(outputs, 0)
    
    y_true_classes = [[] for _ in range(n_classes)]
    y_score_classes = [[] for _ in range(n_classes)]
    for predicted_class, correctness, score in zip(predictions, y_true, y_score):
        y_true_classes[predicted_class].append(correctness)
        y_score_classes[predicted_class].append(score)
        
    if outliers_outputs is not None:
        predictions_outliers = [np.argmax(o) for o in outputs]
        y_score_outliers = rejection_score(outliers_outputs, 0)
        for predicted_class, score in zip(predictions_outliers, y_score_outliers):
            y_true_classes[predicted_class].append(False)
            y_score_classes[predicted_class].append(score)
        
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_classes[i], y_score_classes[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), outputs.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc




def calc_precision_recall(y, outputs):
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    predictions = [np.argmax(o) for o in outputs]
    y_true = [a==b for a,b in zip(np.array(y), np.array(predictions))]
    
    for i in [0,1,2]:
        y_score = rejection_score(outputs, i)
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true, y_score)
        average_precision[i] = metrics.average_precision_score(y_true, y_score)
        
    return precision, recall, average_precision