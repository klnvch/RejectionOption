'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from DataUtils import count_distribution

DATA_DIR = '/home/anton/Desktop/diploma_text/datasets/'

"""
Three gaussian distributed classes, two non-separable and one separable
    add_noise=None - no outliers
    add_noise=1    - just display and return
    add_noise=2    - add noise as no class
    add_noise=3    - add noise as additional class
"""
def create_gaussian_data_2(size=100, add_noise=None, max_output=1.0, min_output=0.0, noise_output=0.0, graphics=False):
    class_1 = np.random.multivariate_normal([ 0.2,  0.2], [[0.04, 0.02], [0.02, 0.04]], size)
    class_2 = np.random.multivariate_normal([-0.2, -0.2], [[0.04, 0.01], [0.01, 0.04]], size)
    class_3 = np.random.multivariate_normal([ 0.6, -0.6], [[0.04, 0.0 ], [0.0 , 0.04]], size)
    noise = None
    if add_noise is not None:
        noise = np.random.uniform(-2, 2, [size, 2])
    
    if graphics:
        lbl_noise = None
        if add_noise is not None:
            lbl_noise,  = plt.plot(*zip(*noise),   marker='s', ls='', label='Outliers',   ms='5' ,color='black')
        lbl_class1, = plt.plot(*zip(*class_1), marker='o', ls='', label='Class 1', ms='5', color='red')
        lbl_class2, = plt.plot(*zip(*class_2), marker='o', ls='', label='Class 2', ms='5', color='green')
        lbl_class3, = plt.plot(*zip(*class_3), marker='p', ls='', label='Class 3', ms='5', color='blue')
        
        if add_noise is not None:
            plt.legend(handles=[lbl_noise, lbl_class1, lbl_class2, lbl_class3], numpoints=1, loc=2)
        else:
            plt.legend(handles=[lbl_class1, lbl_class2, lbl_class3], numpoints=1, loc=2)
        plt.axis([-1, 1, -1, 1])
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.show()
    
    if add_noise is None:
        a = np.concatenate([class_1, class_2, class_3])
        b = np.concatenate([[[max_output, min_output, min_output]] * size, 
                            [[min_output, max_output, min_output]] * size, 
                            [[min_output, min_output, max_output]] * size])
        return a, b, None
    elif add_noise==1:
        a = np.concatenate([class_1, class_2, class_3])
        b = np.concatenate([[[max_output, min_output, min_output]] * size, 
                            [[min_output, max_output, min_output]] * size, 
                            [[min_output, min_output, max_output]] * size])
        return a, b, noise
    elif add_noise==2:
        a = np.concatenate([class_1, class_2, class_3, noise])
        b = np.concatenate([[[max_output,   min_output,   min_output]]   * size, 
                            [[min_output,   max_output,   min_output]]   * size, 
                            [[min_output,   min_output,   max_output]]   * size,
                            [[noise_output, noise_output, noise_output]] * size])
        return a, b, noise
    elif add_noise==3:
        a = np.concatenate([class_1, class_2, class_3, noise])
        b = np.concatenate([[[max_output, min_output, min_output, min_output]] * size, 
                            [[min_output, max_output, min_output, min_output]] * size, 
                            [[min_output, min_output, max_output, min_output]] * size,
                            [[min_output, min_output, min_output, max_output]] * size])
        return a, b, noise
    else:
        assert False
        

"""
Two gaussian linear non-separable classes
"""
def create_gaussian_data_3(size=100, add_noise=None, max_output=1.0, min_output=0.0, noise_output=0.0, graphics=False):
    class_1_part_1 = np.random.multivariate_normal([ 0.4,  0.4], [[0.04, 0.0], [0.0, 0.04]], int(size/2))
    class_1_part_2 = np.random.multivariate_normal([-0.4, -0.4], [[0.04, 0.0], [0.0, 0.04]], int(size/2))
    class_1 = np.concatenate([class_1_part_1, class_1_part_2])
    class_2 = np.random.multivariate_normal([0.2, -0.2], [[0.04, 0], [0, 0.04]], size)
    noise = None
    if add_noise is not None:
        noise = np.random.uniform(-1, 1, [size, 2])
    
    if graphics:
        lbl_noise = None
        if add_noise is not None:
            lbl_noise,  = plt.plot(*zip(*noise),   marker='s', ls='', label='Outliers',   ms='5' ,color='black')
        lbl_class1, = plt.plot(*zip(*class_1), marker='o', markersize='5', color='red',   ls='', label='Class 1')
        lbl_class2, = plt.plot(*zip(*class_2), marker='o', markersize='5', color='green', ls='', label='Class 2')
        
        if add_noise is not None:
            plt.legend(handles=[lbl_noise, lbl_class1, lbl_class2], numpoints=1, loc=2)
        else:
            plt.legend(handles=[lbl_class1, lbl_class2], numpoints=1, loc=2)
        plt.axis([-1, 1, -1, 1])
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.show()
        
    if add_noise is None or not add_noise:
        a = np.concatenate([class_1, class_2])
        b = np.concatenate([[[max_output, min_output]] * size, 
                            [[min_output, max_output]] * size])
        return a, b, noise
    else:
        a = np.concatenate([class_1, class_2, noise])
        b = np.concatenate([[[max_output, min_output]] * size, 
                            [[min_output, max_output]] * size, 
                            [[noise_output, noise_output]] * size])
        return a, b, noise

def read_letter_recognition_image_data():
    x = []
    y = []
    classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    with open(DATA_DIR + 'Letter recognition/letter-recognition.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            x.append([float(i) for i in currentline[1:17]])
            y.append(classes.index(currentline[0]))
            
    return x, y, 26

def read_iris_data():
    iris = datasets.load_iris()
    return iris.data, iris.target, 3

def read_glass_identification_data():
    x = []
    y = []
    classes = [1,2,3,5,6,7]
    with open(DATA_DIR + 'Glass identification/glass.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            x.append([float(i) for i in currentline[1:10]])
            y.append(classes.index(int(currentline[10])))
            
    return x, y, 6

def read_image_segmentation_data():
    x = []
    y = []
    classes = ['BRICKFACE','SKY','FOLIAGE','CEMENT','WINDOW','PATH','GRASS']
    with open(DATA_DIR + 'Image segmentation/segmentation.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 20: continue
            x.append([float(i) for i in currentline[1:20]])
            y.append(classes.index(currentline[0]))
            
    with open(DATA_DIR + 'Image segmentation/segmentation.test', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 20: continue
            x.append([float(i) for i in currentline[1:20]])
            y.append(classes.index(currentline[0]))
    
    return x, y, 7

def get_data(i, binarize=False, preprocess=0):
    """Returns choosen dataset.

    Args:
        i: index of the dataset.
        
        1 - Letter recognition image data
        2 - Iris data
        3 - Glass identification data
        4 - Image segmentation data 
    Returns:
        two arrays data and labels.
    Raises:
        ValueError: unknown error with data.

    """
    if   i == 1: x, y, n_classes = read_letter_recognition_image_data()
    elif i == 2: x, y, n_classes = read_iris_data()
    elif i == 3: x, y, n_classes = read_glass_identification_data()
    elif i == 4: x, y, n_classes = read_image_segmentation_data()
    else: raise ValueError("unknown dataset: " + i)
    
    if binarize:
        y = label_binarize(y, range(0, n_classes), 0, 1, False)
        count_distribution(y)
    
    if preprocess == 1:
        x = preprocessing.scale(x)
    elif preprocess == 2:
        x = preprocessing.minmax_scale(x)
    elif preprocess == 3:
        x = preprocessing.normalize(x)
    elif preprocess == 4:
        x = preprocessing.robust_scale(x)
    
    return np.array(x), np.array(y), n_classes