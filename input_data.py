'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from data_utils import count_distribution
from collections import Counter
from graphics import plot_2d_dataset

DATA_DIR = 'datasets/'

def read_letter_recognition_image_data():
    x = []
    y = []
    classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    with open(DATA_DIR + 'Letter recognition/letter-recognition.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            x.append([np.float32(i) for i in currentline[1:17]])
            y.append(classes.index(currentline[0]))
            
    return x, y, classes

def read_iris_data():
    iris = datasets.load_iris()
    return iris.data, iris.target, ['Setosa', 'Versicolour', 'Virginica']

def read_glass_identification_data():
    x = []
    y = []
    classes = [1,2,3,5,6,7]
    with open(DATA_DIR + 'Glass identification/glass.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            x.append([np.float32(i) for i in currentline[1:10]])
            y.append(classes.index(int(currentline[10])))
            
    return x, y, ['building windows float processed', 
                  'building windows non float processed', 
                  'vehicle windows float processed', 
                  'containers', 'tableware', 'headlamps']

def read_image_segmentation_data():
    x = []
    y = []
    classes = ['BRICKFACE','SKY','FOLIAGE','CEMENT','WINDOW','PATH','GRASS']
    with open(DATA_DIR + 'Image segmentation/segmentation.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 20: continue
            x.append([np.float32(i) for i in currentline[1:20]])
            y.append(classes.index(currentline[0]))
            
    with open(DATA_DIR + 'Image segmentation/segmentation.test', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 20: continue
            x.append([np.float32(i) for i in currentline[1:20]])
            y.append(classes.index(currentline[0]))
    
    return x, y, classes

def generate_multiclass():
    x, y = datasets.make_classification(n_samples=1000,
                                        n_features=2,
                                        n_informative=2,
                                        n_redundant=0,
                                        n_repeated=0,
                                        n_classes=3,
                                        n_clusters_per_class=1,
                                        weights=None,
                                        flip_y=0.01,
                                        class_sep=1.0,
                                        hypercube=True,
                                        shift=0.0,
                                        scale=1.0,
                                        shuffle=True,
                                        random_state=None)
    return x, y, ['1', '2', '3']

def generate_blobs():
    x, y = datasets.make_blobs(n_samples=1000, 
                               n_features=2, 
                               centers=8, 
                               cluster_std=1.0, 
                               center_box=(-10.0, 10.0), 
                               shuffle=True, 
                               random_state=None)
    return x, y, ['0', '1', '2', '3', '4', '5', '6', '7']

def generate_circles():
    x, y = datasets.make_circles(n_samples=1000, 
                                 shuffle=True, 
                                 noise=0.2, 
                                 random_state=None, 
                                 factor=.5)
    return x, y, ['0', '1']

def generate_moons():
    x, y = datasets.make_moons(n_samples=1000, 
                               shuffle=True, 
                               noise=0.3, 
                               random_state=None)
    return x, y, ['0', '1']

def generate_quantiles(mode=1):
    x, y = datasets.make_gaussian_quantiles(mean=None, 
                                            cov=1., 
                                            n_samples=1000, 
                                            n_features=2, 
                                            n_classes=3, 
                                            shuffle=True, 
                                            random_state=None)
    if mode == 1:
        return x, y, ['0', '1', '2']
    else:
        y[y==2] = 0
        return x, y, ['0', '1']

def get_data(i, binarize=False, preprocess=0):
    """Returns choosen dataset.

    Args:
        i: index of the dataset.
            1 - Letter recognition image data
            2 - Iris data
            3 - Glass identification data
            4 - Image segmentation data
            5 - Moons
            6 - Blobs
            7 - Circles
            8 - Quantiles
            9 - Multiclass
        preprocess: index of algorithm
            1 - scale
            2 - minmax scale
            3 - normilize
            4 - robust scale
    Returns:
        features, output, classes 
    Raises:
        ValueError: unknown error with data.

    """
    if   i == 1: x, y, classes = read_letter_recognition_image_data()
    elif i == 2: x, y, classes = read_iris_data()
    elif i == 3: x, y, classes = read_glass_identification_data()
    elif i == 4: x, y, classes = read_image_segmentation_data()
    elif i == 5: x, y, classes = generate_moons()
    elif i == 6: x, y, classes = generate_blobs()
    elif i == 7: x, y, classes = generate_circles()
    elif i == 8: x, y, classes = generate_quantiles(mode=2)
    elif i == 9: x, y, classes = generate_multiclass()
    else: raise ValueError("unknown dataset: " + i)
    
    if binarize:
        if len(classes) == 2:
            y = preprocessing.label_binarize(y, range(len(classes)+1), 0, 1, False)[:,:-1]
        else:
            y = preprocessing.label_binarize(y, range(len(classes)), 0, 1, False)
        count_distribution(y)
        
    if preprocess == 1:
        x = preprocessing.scale(x)
    elif preprocess == 2:
        x = preprocessing.minmax_scale(x, feature_range=(0, 1))
    elif preprocess == 3: # normilize row a = a / sqrt(a+b+...)
        x = preprocessing.normalize(x)
    elif preprocess == 4:
        x = preprocessing.robust_scale(x)
    
    return np.array(x), np.array(y), classes

def print_stats(x, clases):
    print(x.min(axis=0))
    print(x.max(axis=0))
    print(x.mean(axis=0))

def print_classes_stats(y, classes):
    """
    Helper for generating class distibution table in chapter 1
    """
    print(classes)
    print('&'.join(classes))
    dist = np.array(list(Counter(y).values()))
    print(dist)
    print('&'.join(map(str,dist)))
    dist = 100 * dist / len(y)
    print(dist)
    print('&'.join(map(str,dist)))
    
if __name__ == '__main__':
    x, y, _ = generate_multiclass()
    plot_2d_dataset(x, y)