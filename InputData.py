'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt

"""
Three gaussian distributed classes, two non-separable and one separable
    add_noise=None - no outliers
    add_noise=1    - just display and return
    add_noise=2    - add noise to the classes with minimal output
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
    a = []
    b = []
    with open("/home/anton/Desktop/diploma/data sets/Letter recognition/letter-recognition.data", "r") as filestream:
        for line in filestream:
            currentline = line.split(',')
            
            a.append([int(i) for i in currentline[1:17]])
            
            if   currentline[0]== 'A': b.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'B': b.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'C': b.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'D': b.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'E': b.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'F': b.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'G': b.append([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'H': b.append([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'I': b.append([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'J': b.append([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'K': b.append([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'L': b.append([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'M': b.append([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'N': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'O': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'P': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'Q': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'R': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
            elif currentline[0]== 'S': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
            elif currentline[0]== 'T': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
            elif currentline[0]== 'U': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
            elif currentline[0]== 'V': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
            elif currentline[0]== 'W': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
            elif currentline[0]== 'X': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
            elif currentline[0]== 'Y': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
            elif currentline[0]== 'Z': b.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
            else: raise ValueError("unknown class: " + currentline[0])
    return np.array(a), np.array(b)

def read_iris_data():
    a = []
    b = []
    with open("/home/anton/Desktop/diploma/data sets/Iris/iris.data", "r") as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 5: continue

            a.append([float(i) for i in currentline[0:4]])
            
            if   currentline[4].startswith('Iris-setosa'):     b.append([1, 0, 0])
            elif currentline[4].startswith('Iris-versicolor'): b.append([0, 1, 0])
            elif currentline[4].startswith('Iris-virginica'):  b.append([0, 0, 1])
            else: raise ValueError("unknown class: " + currentline[4])
    return np.array(a), np.array(b)

def read_glass_identification_data():
    a = []
    b = []
    with open('/home/anton/Desktop/diploma/data sets/Glass identification/glass.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            
            a.append([float(i) for i in currentline[1:10]])
            
            if   int(currentline[10])== 1: b.append([1,0,0,0,0,0])
            elif int(currentline[10])== 2: b.append([0,1,0,0,0,0])
            elif int(currentline[10])== 3: b.append([0,0,1,0,0,0])
            elif int(currentline[10])== 5: b.append([0,0,0,1,0,0])
            elif int(currentline[10])== 6: b.append([0,0,0,0,1,0])
            elif int(currentline[10])== 7: b.append([0,0,0,0,0,1])
            else: raise ValueError("unknown class: " + currentline[4])
    return np.array(a), np.array(b)

def read_image_segmentation_data():
    a = []
    b = []
    with open('/home/anton/Desktop/diploma/data sets/Image segmentation/segmentation.data', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 20: continue
            
            a.append([float(i) for i in currentline[1:20]])
            
            if   currentline[0]== 'BRICKFACE': b.append([1,0,0,0,0,0,0])
            elif currentline[0]== 'SKY':       b.append([0,1,0,0,0,0,0])
            elif currentline[0]== 'FOLIAGE':   b.append([0,0,1,0,0,0,0])
            elif currentline[0]== 'CEMENT':    b.append([0,0,0,1,0,0,0])
            elif currentline[0]== 'WINDOW':    b.append([0,0,0,0,1,0,0])
            elif currentline[0]== 'PATH':      b.append([0,0,0,0,0,1,0])
            elif currentline[0]== 'GRASS':     b.append([0,0,0,0,0,0,1])
            else: raise ValueError("unknown class: " + currentline[0])
    with open('/home/anton/Desktop/diploma/data sets/Image segmentation/segmentation.test', 'r') as filestream:
        for line in filestream:
            currentline = line.split(',')
            if len(currentline) != 20: continue
            
            a.append([float(i) for i in currentline[1:20]])
            
            if   currentline[0]== 'BRICKFACE': b.append([1,0,0,0,0,0,0])
            elif currentline[0]== 'SKY':       b.append([0,1,0,0,0,0,0])
            elif currentline[0]== 'FOLIAGE':   b.append([0,0,1,0,0,0,0])
            elif currentline[0]== 'CEMENT':    b.append([0,0,0,1,0,0,0])
            elif currentline[0]== 'WINDOW':    b.append([0,0,0,0,1,0,0])
            elif currentline[0]== 'PATH':      b.append([0,0,0,0,0,1,0])
            elif currentline[0]== 'GRASS':     b.append([0,0,0,0,0,0,1])
            else: raise ValueError("unknown class: " + currentline[0])
    return np.array(a), np.array(b)

def get_data(i):
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
    if i == 1: return read_letter_recognition_image_data()
    elif i == 2: return read_iris_data()
    elif i == 3: return read_glass_identification_data()
    elif i == 4: return read_image_segmentation_data()
    else: raise ValueError("unknown dataset: " + i)
    
    