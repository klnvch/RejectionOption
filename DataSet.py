'''
Created on Apr 13, 2017

@author: anton

Class for preparation dataset for ANN
'''
import copy

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

from input_data import get_data, read_marcin_file
import numpy as np


class Set:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = x.shape[0]
        self.n_features = x.shape[1]
        self.n_classes = y.shape[1]
    
    def add_noise(self, x=None, noise_size=1.0):
        if x is None:
            noise_size = int(self.size * noise_size)
            x = np.random.uniform(self.x.min() - 1, self.x.max() + 1,
                                        [noise_size, self.n_features])
        else:
            noise_size = x.shape[0]
        
        noise_output = self.y.min()
        y = np.array([[noise_output] * self.n_classes] * noise_size)
        self.x = np.concatenate([self.x, x])
        self.y = np.concatenate([self.y, y])
        
        self.size = self.y.shape[0]
        self.n_classes = self.y.shape[1]
    
    def add_class(self, x=None, noise_size=1.0):
        if x is None:
            noise_size = int(self.size * noise_size)
            x = np.random.uniform(self.x.min() - 1, self.x.max() + 1,
                                  [noise_size, self.n_features])
        else:
            noise_size = x.shape[0]
        
        y = np.zeros((noise_size, self.n_classes + 1))
        y[:, -1] = 1
        
        self.x = np.concatenate([self.x, x])
        self.y = np.append(self.y, np.zeros((self.size, 1)), axis=1)
        self.y = np.concatenate([self.y, y])
        
        self.size = self.y.shape[0]
        self.n_classes = self.y.shape[1]
    
    def remove_class(self, i):
        new_x = []
        new_y = []
        outliers = []
        for _x, _y in zip(self.x, self.y):
            if _y.argmax() == i:
                outliers.append(_x)
            else:
                new_x.append(_x)
                new_y.append(np.delete(_y, i))
        
        self.x = np.array(new_x)
        self.y = np.array(new_y)
        self.size = self.x.shape[0]
        self.n_classes = self.y.shape[1]
        
        return np.array(outliers)

class DataSet:
    
    def __init__(self, ds_id, n_samples=1000, split=[0.6, 0.1, 0.3],
                 add_noise=None, noise_size=1.0, noise_output=0.0,
                 output=None, random_state=None):
        """
        Args:
            ds_id: index of the dataset.
                1 - Letter recognition image data
                2 - Iris data
                3 - Glass identification data
                4 - Image segmentation data
                
                5 - Moons Overlapping
                6 - Blobs
                7 - Circles
                8 - Gaussian Quantiles modified
                9 - Multiclass
                10 - Gaussian Quantiles
                11 - Noise
                12 - Moons Separable
                
                13 - Marcin Luckner Dataset
                14 - MNIST Dataset
            
            add_noise: noise type
                0: no outliers
                1: add noise as no class
                2: add noise as a class
                3: only outliers
                
                5: add outliers as no class
                6: add outliers as a class
                
                8: remove class
        """
        if ds_id == 13:
            self.load_marcin_dataset(add_noise, output)
            return
        elif ds_id == 14:
            self.load_mnist_data(add_noise)
            return
        
        # load data
        x, y, self.target_names = get_data(ds_id, n_samples, binarize=True,
                                           random_state=random_state)
        self.n_features = x.shape[1]
        self.n_classes = y.shape[1]
        self.outliers = None
        # split into 60%, 10% and 30%
        if len(split) == 3:
            trn_size = split[0]
            vld_size = split[1]
            tst_size = split[2]
            
            x_trn, x_tst, y_trn, y_tst = \
                train_test_split(x, y, train_size=trn_size)
            x_vld, x_tst, y_vld, y_tst = \
                train_test_split(x_tst, y_tst, train_size=vld_size / (vld_size + tst_size))
            
            self.trn = Set(x_trn, y_trn)
            self.vld = Set(x_vld, y_vld)
            self.tst = Set(x_tst, y_tst)
        elif len(split) == 2:
            splt1 = train_test_split(x, y, test_size=split[0])
            self.trn = Set(splt1[1], splt1[3])
            self.vld = None
            self.tst = Set(splt1[0], splt1[2])
        
        self.print_info()
        
        # preprocess
        self.scale()
        
        # add noise
        if add_noise == 1:
            self.add_noise_as_no_class(noise_size, noise_output)
        elif add_noise == 2:
            self.add_noise_as_a_class()
        elif add_noise == 3:
            self.outliers = np.random.uniform(self.tst.x.min() - 1,
                                          self.tst.x.max() + 1,
                                          [self.tst.size, self.n_features])
        self.print_info()
    
    def copy(self):
        return copy.deepcopy(self)
    
    def print_info(self):
        if self.vld is None:
            print('{:d}|{:d}'.format(self.trn.size, self.tst.size))
        else:
            print('trn: {:d}, vld: {:d}, tst: {:d}'.format(self.trn.size,
                                                           self.vld.size,
                                                           self.tst.size))
        if self.outliers is not None:
            print('outliers: {:s}'.format(str(self.outliers.shape)))
    
    def add_noise_as_no_class(self, noise_size=1.0, noise_output=None):
        """
        Adds noise with low output to the training set
        Adds outliers set
        Args:
            noise_size: numner of noise patterns, default is side of the dataset
            noise_output: noise output, defult is 1./number of classes
        Returns:
            new dataset, ds_x,ds_y and outliers
        """
        self.trn.add_noise(noise_size)
        self.outliers = np.random.uniform(self.tst.x.min() - 1,
                                          self.tst.x.max() + 1,
                                          [self.tst.size * 4, self.n_features])
        
        self.print_info()
    
    def add_noise_as_a_class(self, noise_size=1.0):
        """
        Adds noise as a class to a dataset
        Args:
            noise_size: numner of noise patterns, default is side of the dataset
        Returns:
            new dataset, ds_x,ds_y and outliers
        """
        
        self.trn.add_noise_class(noise_size)
        if self.vld is not None: self.vld.add_noise_class(noise_size)
        # self.tst.add_noise_class(1.0)
        self.tst.y = np.append(self.tst.y, np.zeros((self.tst.size, 1)), axis=1)
        self.outliers = np.random.uniform(self.tst.x.min() - 1,
                                          self.tst.x.max() + 1,
                                          [self.tst.size * 4, self.n_features])
        
        self.target_names = np.concatenate([self.target_names, ['Outliers']])
        self.n_classes += 1
        
        self.print_info()
    
    def remove_class(self, i):
        """
        Removes class of index i in the dataset
        Args:
            u: a class to be removed and added to outliers
        """
        out_1 = self.trn.remove_class(i)
        out_2 = self.vld.remove_class(i)
        out_3 = self.tst.remove_class(i)
        
        self.outliers = np.concatenate([out_1, out_2, out_3])
        self.n_classes -= 1
        self.target_names = np.delete(self.target_names, i)
        
        self.print_info()
    
    def change_targets(self, targets):
        y = self.trn.y
        
        neg_label, pos_label = targets
        y = y.astype(float)
        y[y == 0.0] = neg_label
        y[y == 1.0] = pos_label
    
    def pca(self):
        pca = PCA(n_components=self.n_features).fit(self.trn.x)
        self.trn.x = pca.transform(self.trn.x)
        if self.vld is not None:
            self.vld.x = pca.transform(self.vld.x)
        self.tst.x = pca.transform(self.tst.x)
        if self.outliers is not None:
            self.outliers = pca.transform(self.outliers)
    
    def scale(self):
        scaler = preprocessing.StandardScaler().fit(self.trn.x)
        self.trn.x = scaler.transform(self.trn.x)
        if self.vld is not None:
            self.vld.x = scaler.transform(self.vld.x)
        self.tst.x = scaler.transform(self.tst.x)
        if self.outliers is not None:
            self.outliers = scaler.transform(self.outliers)
    
    def load_marcin_dataset(self, add_noise, output):
        """
        - load data
        - binarize output
        - load outliers if nesessary
        - add outliers to the training set if nessasary
        - scale
        - create outliers if nessesary
        """
        # load data
        x_trn, y_trn, self.target_names = read_marcin_file('LirykaLearning.csv')
        x_vld, y_vld, _ = read_marcin_file('LirykaValidate.csv')
        x_tst, y_tst, _ = read_marcin_file('LirykaTesting.csv')
        
        self.n_features = x_trn.shape[1]
        self.n_classes = self.target_names.shape[0]
        self.outliers = None
        
        y_trn = preprocessing.label_binarize(y_trn, range(self.n_classes))
        y_vld = preprocessing.label_binarize(y_vld, range(self.n_classes))
        y_tst = preprocessing.label_binarize(y_tst, range(self.n_classes))
        
        self.trn = Set(x_trn, y_trn)
        self.vld = Set(x_vld, y_vld)
        self.tst = Set(x_tst, y_tst)
        
        if add_noise in [1, 2, 3, 5, 6]:
            # init outliers
            o_trn_1, _, _ = read_marcin_file('AccidentalsTesting.csv')
            o_trn_2, _, _ = read_marcin_file('DynamicsTesting.csv')
            o_trn_3, _, _ = read_marcin_file('RestsTesting.csv')
            self.outliers = np.concatenate((o_trn_1, o_trn_2, o_trn_3))
        
        if add_noise in [5, 6]:
            o_trn_1, _, _ = read_marcin_file('AccidentalsLearning.csv')
            o_trn_2, _, _ = read_marcin_file('DynamicsLearning.csv')
            o_trn_3, _, _ = read_marcin_file('RestsLearning.csv')
            o_trn = np.concatenate((o_trn_1, o_trn_2, o_trn_3))
            
            if add_noise == 5:
                self.trn.add_noise(o_trn)
            else:
                self.trn.add_class(o_trn)
                self.vld.add_class(np.empty((0, self.n_features)))
                self.tst.add_class(np.empty((0, self.n_features)))
        
        self.scale()
        
        if add_noise in [1, 2]:
            if add_noise == 1:
                self.trn.add_noise()
            else:
                self.trn.add_class()
                self.vld.add_class(np.empty((0, self.n_features)))
                self.tst.add_class(np.empty((0, self.n_features)))
        
        if add_noise in [2, 6]:
            self.n_classes += 1
            self.target_names = np.concatenate((self.target_names,
                                                ['Outliers']))
        
        if output is not None:
            self.change_targets(output)
        
        self.print_info()
    
    def load_mnist_data(self, add_noise=None):
        mnist = input_data.read_data_sets("datasets/MNIST/", one_hot=True)
        
        x_trn, y_trn = mnist.train.images, mnist.train.labels
        x_vld, y_vld = mnist.validation.images, mnist.validation.labels
        x_tst, y_tst = mnist.test.images[:8000], mnist.test.labels[:8000]
        
        self.target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.n_features = x_trn.shape[1]
        self.n_classes = 10
        self.outliers = None
        
        self.trn = Set(x_trn, y_trn)
        self.vld = Set(x_vld, y_vld)
        self.tst = Set(x_tst, y_tst)
        
        self.print_info()
        
        if add_noise == 8:
            self.remove_class(6)
