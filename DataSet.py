'''
Created on Apr 13, 2017

@author: anton

Class for preparation dataset for ANN
'''
import numpy as np
from input_data import get_data
from sklearn.model_selection import train_test_split

class Set:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = x.shape[0]
        self.n_features = x.shape[1]
        self.n_classes = y.shape[1]
    
    def add_noise_class(self, noise_size=None):
        if noise_size is None: noise_size = self.size
        
        noise_x = np.random.uniform(self.x.min()-1, self.x.max()+1,
                                    [noise_size, self.n_features])
        noise_y = np.array([[0]*(self.n_classes) + [1]] * noise_size)
        
        self.x = np.concatenate([self.x, noise_x])
        self.y = np.append(self.y, np.array([[0]] * self.size), axis=1)
        self.y = np.concatenate([self.y, noise_y])
        
        self.size = self.y.shape[0]
        self.n_classes = self.y.shape[1]

class DataSet:
    
    def __init__(self, dataset, size=1000, split=[0.6, 0.2, 0.2], 
                 add_noise=None, noise_output=0.0):
        """
        Args:
            dataset: index of the dataset.
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
            add_noise: noise type
                1:
        """
        # load data
        x, y, self.target_names = get_data(dataset, size,
                                           binarize=True, preprocess=1)
        self.n_features = x.shape[1]
        self.n_classes = y.shape[1]
        self.outliers = None
        # split into 60%, 20% and 20%
        if len(split) == 3:
            splt1 = train_test_split(x, y, test_size=split[0])
            splt2 = train_test_split(splt1[0], splt1[2],
                                     test_size=split[1] / (1.0 - split[0]))
            self.trn = Set(splt1[1], splt1[3])
            self.vld = Set(splt2[1], splt2[3])
            self.tst = Set(splt2[0], splt2[2])
            print('{:d}|{:d}|{:d}'.format(self.trn.size, self.vld.size, 
                                          self.tst.size))
        elif len(split) == 2:
            splt1 = train_test_split(x, y, test_size=split[0])
            self.trn = Set(splt1[1], splt1[3])
            self.vld = None
            self.tst = Set(splt1[0], splt1[2])
            print('{:d}|{:d}'.format(self.trn.size, self.tst.size))
        # add noise
        if add_noise == 1:
            self.add_noise_as_no_class(None, noise_output)
        elif add_noise == 2:
            self.add_noise_as_a_class(None)
        elif add_noise == 3:
            self.outliers = np.random.uniform(self.tst.x.min()-1,
                                          self.tst.x.max()+1,
                                          [self.tst.size, self.n_features])
        # print final sizes
        if self.vld is None:
            print('{:d}|{:d}'.format(self.trn.size, self.tst.size))
        else:
            print('{:d}|{:d}|{:d}'.format(self.trn.size, self.vld.size, 
                                          self.tst.size))
    
    def add_noise_as_no_class(self, noise_size=None, noise_output=None):
        """
        Adds noise with low output to the training set
        Adds outliers set
        Args:
            noise_size: numner of noise patterns, default is side of the dataset
            noise_output: noise output, defult is 1./number of classes
        Returns:
            new dataset, ds_x,ds_y and outliers
        """
        if noise_size is None: noise_size = self.trn.size
        if noise_output is None: noise_output = self.trn.y.min()
        
        noise_x = np.random.uniform(self.trn.x.min()-1, self.trn.x.max()+1,
                                    [noise_size, self.n_features])
        noise_y = np.array([[noise_output] * self.n_classes] * noise_size)
        
        new_x = np.concatenate([self.trn.x, noise_x])
        new_y = np.concatenate([self.trn.y, noise_y])
        
        self.trn = Set(new_x, new_y)
        self.outliers = np.random.uniform(self.tst.x.min()-1,
                                          self.tst.x.max()+1,
                                          [self.tst.size, self.n_features])
        
    def add_noise_as_a_class(self, noise_size=None):
        """
        Adds noise as a class to a dataset
        Args:
            noise_size: numner of noise patterns, default is side of the dataset
        Returns:
            new dataset, ds_x,ds_y and outliers
        """
        
        self.trn.add_noise_class(noise_size)
        if self.vld is not None: self.vld.add_noise_class(noise_size)
        self.tst.add_noise_class(noise_size)
        self.target_names = np.concatenate([self.target_names, ['Outliers']])
        self.n_classes += 1