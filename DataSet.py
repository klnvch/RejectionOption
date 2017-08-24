'''
Created on Apr 13, 2017

@author: anton

Class for preparation dataset for ANN
'''
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from input_data import get_data, read_marcin_file
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

class Set:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = x.shape[0]
        self.n_features = x.shape[1]
        self.n_classes = y.shape[1]
    
    def add_noise_class(self, noise_size=1.0):
        noise_size = int(self.size * noise_size)
        
        noise_x = np.random.uniform(self.x.min()-1, self.x.max()+1,
                                    [noise_size, self.n_features])
        noise_y = np.array([[0]*(self.n_classes) + [1]] * noise_size)
        
        self.x = np.concatenate([self.x, noise_x])
        self.y = np.append(self.y, np.array([[0]] * self.size), axis=1)
        self.y = np.concatenate([self.y, noise_y])
        
        self.size = self.y.shape[0]
        self.n_classes = self.y.shape[1]

class DataSet:
    
    def __init__(self, ds_id, size=1000, split=[0.6, 0.2, 0.2], 
                 add_noise=None, noise_size=1.0, noise_output=0.0,
                 output=None):
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
            
            add_noise: noise type
                1: add noise as no class
        """
        if ds_id == 13:
            self.load_marcin_dataset(add_noise, output)
            return
        
        # load data
        x, y, self.target_names = get_data(ds_id, size, binarize=True)
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
        elif len(split) == 2:
            splt1 = train_test_split(x, y, test_size=split[0])
            self.trn = Set(splt1[1], splt1[3])
            self.vld = None
            self.tst = Set(splt1[0], splt1[2])
        
        self.print_info()
        # add noise
        if add_noise == 1:
            self.add_noise_as_no_class(noise_size, noise_output)
        elif add_noise == 2:
            self.add_noise_as_a_class(None)
        elif add_noise == 3:
            self.outliers = np.random.uniform(self.tst.x.min()-1,
                                          self.tst.x.max()+1,
                                          [self.tst.size, self.n_features])
        
        # preprocess
        self.scale()
    
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
        noise_size = int(self.trn.size * noise_size)
        if noise_output is None: noise_output = self.trn.y.min()
        
        noise_x = np.random.uniform(self.trn.x.min()-1, self.trn.x.max()+1,
                                    [noise_size, self.n_features])
        noise_y = np.array([[noise_output] * self.n_classes] * noise_size)
        
        new_x = np.concatenate([self.trn.x, noise_x])
        new_y = np.concatenate([self.trn.y, noise_y])
        
        self.trn = Set(new_x, new_y)
        self.outliers = np.random.uniform(self.tst.x.min()-1,
                                          self.tst.x.max()+1,
                                          [self.tst.size*4, self.n_features])
        
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
        self.tst.add_noise_class(1.0)
        self.target_names = np.concatenate([self.target_names, ['Outliers']])
        self.n_classes += 1
        
        self.print_info()
        
    def change_targets(self, targets):
        y = self.trn.y
        
        neg_label, pos_label = targets
        y = y.astype(float)
        y[y==0.0] = neg_label
        y[y==1.0] = pos_label
    
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
        # load data
        x_trn, y_trn, self.target_names = read_marcin_file('LirykaLearning.csv')
        x_vld, y_vld, _ = read_marcin_file('LirykaValidate.csv')
        x_tst, y_tst, _ = read_marcin_file('LirykaTesting.csv')
        
        self.n_features = x_trn.shape[1]
        self.n_classes = self.target_names.shape[0]
        self.outliers = None
        
        if add_noise == 2:
            o_trn_1, _, _ = read_marcin_file('AccidentalsLearning.csv')
            o_trn_2, _, _ = read_marcin_file('DynamicsLearning.csv')
            o_trn_3, _, _ = read_marcin_file('RestsLearning.csv')
            o_trn = np.concatenate((o_trn_1, o_trn_2, o_trn_3))
            
            x_trn = np.concatenate((x_trn, o_trn))
            y_trn = np.concatenate((y_trn,
                                    np.array([self.n_classes]*o_trn.shape[0])))
            
            
            o_trn_1, _, _ = read_marcin_file('AccidentalsTesting.csv')
            o_trn_2, _, _ = read_marcin_file('DynamicsTesting.csv')
            o_trn_3, _, _ = read_marcin_file('RestsTesting.csv')
            o_trn = np.concatenate((o_trn_1, o_trn_2, o_trn_3))
            
            x_tst = np.concatenate((x_tst, o_trn))
            y_tst = np.concatenate((y_tst,
                                    np.array([self.n_classes]*o_trn.shape[0])))
            
            self.n_classes += 1
            self.target_names = np.concatenate((self.target_names, ['Outliers']))
        
        y_trn = preprocessing.label_binarize(y_trn, range(self.n_classes))
        y_vld = preprocessing.label_binarize(y_vld, range(self.n_classes))
        y_tst = preprocessing.label_binarize(y_tst, range(self.n_classes))
        
        if add_noise == 1:
            o_trn_1, _, _ = read_marcin_file('AccidentalsLearning.csv')
            o_trn_2, _, _ = read_marcin_file('DynamicsLearning.csv')
            o_trn_3, _, _ = read_marcin_file('RestsLearning.csv')
            o_trn = np.concatenate((o_trn_1, o_trn_2, o_trn_3))
            
            x_trn = np.concatenate((x_trn, o_trn))
            y_trn = np.concatenate((y_trn,
                                    np.zeros((o_trn.shape[0], self.n_classes))))
        
        if add_noise == 1 or add_noise == 3:
            o_trn_1, _, _ = read_marcin_file('AccidentalsTesting.csv')
            o_trn_2, _, _ = read_marcin_file('DynamicsTesting.csv')
            o_trn_3, _, _ = read_marcin_file('RestsTesting.csv')
            self.outliers = np.concatenate((o_trn_1, o_trn_2, o_trn_3))
        
        #self.n_features = 64
        #pca = PCA(n_components=self.n_features).fit(x_trn)
        #x_trn = pca.transform(x_trn)
        #x_vld = pca.transform(x_vld)
        #x_tst = pca.transform(x_tst)
        #self.outliers = pca.transform(self.outliers)
        
        #vt = VarianceThreshold(threshold=50).fit(x_trn)
        #x_trn = vt.transform(x_trn)
        #x_vld = vt.transform(x_vld)
        #x_tst = vt.transform(x_tst)
        #self.outliers = vt.transform(self.outliers)
        #self.n_features = x_trn.shape[1]
        
        scaler = preprocessing.StandardScaler().fit(x_trn)
        x_trn = scaler.transform(x_trn)
        x_vld = scaler.transform(x_vld)
        x_tst = scaler.transform(x_tst)
        if self.outliers is not None:
            self.outliers = scaler.transform(self.outliers)

        self.trn = Set(x_trn, y_trn)
        self.vld = Set(x_vld, y_vld)
        self.tst = Set(x_tst, y_tst)
        
        if output is not None:
            self.change_targets(output)
        
        self.print_info()