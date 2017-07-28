'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import csv
from sklearn import datasets
from sklearn import preprocessing
from collections import Counter
from graphics import plot_2d_dataset

DATA_DIR = 'datasets/'

def read_letter_recognition_image_data():
    x = []
    y = []
    classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
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

def read_marcin_file(name):
    with open(DATA_DIR + 'Archiwum/' + name) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        columns = reader.__next__()
        print(columns)
        values = np.array([row for row in reader if len(row) > 0])
        
    # remove class (the first) and empty (the last)  columns
    x = values[:,1:-1]
    x = x.astype(np.float)
    # find all classes from the first column
    classes = np.unique(values[:,0])
    # find indecies of classes for each row
    y = np.array([np.where(classes==c)[0][0] for c in values[:,0]])
    
    return x, y, classes

def print_marcin_data():
    _, y, classes = read_marcin_file('LirykaLearning.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    _, y, classes = read_marcin_file('LirykaValidate.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    _, y, classes = read_marcin_file('LirykaTesting.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    
    _, y, classes = read_marcin_file('AccidentalsLearning.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    _, y, classes = read_marcin_file('AccidentalsTesting.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    
    _, y, classes = read_marcin_file('DynamicsLearning.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    _, y, classes = read_marcin_file('DynamicsTesting.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    
    _, y, classes = read_marcin_file('RestsLearning.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')
    _, y, classes = read_marcin_file('RestsTesting.csv')
    print_classes_stats(y, classes, 'tests/marcin_dataset.csv')

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

def generate_moons(n_samples=1000, noise=0.3):
    """
    Keep noise=0.3  to create two classes with overlapping regions,
    so it can be possible to separate them by three hyperplanes
    
    Keep noise=0.1 to create two classes tha are separable,
    but not linearly non-separable
    """
    x, y = datasets.make_moons(n_samples=n_samples,
                               shuffle=True,
                               noise=noise,
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
    
def generate_noise(size = 200):
    x = np.random.uniform(-1, 1, (size, 2))
    return x, [0]*size, ['0']

def get_data(i, size, binarize=False, preprocess=0):
    """Returns choosen dataset.

    Args:
        i: index of the dataset.
            1 - Letter recognition image data
            2 - Iris data
            3 - Glass identification data
            4 - Image segmentation data
            5 - Moons non-separable
            6 - Blobs
            7 - Circles
            8 - Gaussian Quantiles modified
            9 - Multiclass
            10 - Gaussian Quantiles
            11 - Noise
            12 - Moons separable
            
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
    elif i == 5: x, y, classes = generate_moons(size, 0.3)
    elif i == 6: x, y, classes = generate_blobs()
    elif i == 7: x, y, classes = generate_circles()
    elif i == 8: x, y, classes = generate_quantiles(mode=2)
    elif i == 9: x, y, classes = generate_multiclass()
    elif i == 10: x, y, classes = generate_quantiles(mode=1)
    elif i == 11: x, y, classes = generate_noise()
    elif i == 12: x, y, classes = generate_moons(size, 0.1)
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

def count_distribution(y):
    """
    Check if needed
    """
    d = [0] * y.shape[1]
    
    for i in y:
        d[i.argmax()] += 1
        
    #d = np.asarray(d) / ds_y.shape[0]
    print(d)
    return d

def print_classes_stats(y, classes, path=None):
    """
    Helper for generating class distibution table in chapter 1
    """
    if path is None:
        print(classes)
        print('&'.join(classes))
        dist = np.array(list(Counter(y).values()))
        print(dist)
        print('&'.join(map(str,dist)))
        dist = 100 * dist / len(y)
        print(dist)
        print('&'.join(map(str,dist)))
    else:
        with open(path, 'a+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(classes)
            dist = np.array(list(Counter(y).values()))
            writer.writerow(dist)
            dist = 100 * dist / len(y)
            writer.writerow(dist)

if __name__ == '__main__':
    x, y, _ = generate_moons(1000, 0.1)
    plot_2d_dataset(x, y)
    """
    x, y, classes = read_marcin_file('LirykaLearning.csv')
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=9)
    pca.fit(x)
    print(x.shape)
    x = pca.transform(x)
    print(x.shape)
    print(pca.explained_variance_ratio_)
    print(pca.components_)
    """