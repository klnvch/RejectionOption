'''
Created on Apr 13, 2017

@author: anton

Class for preparation dataset for ANN
'''

from input_data import get_data
from sklearn.model_selection import train_test_split
from data_utils import add_noise_as_no_class, add_noise_as_a_class

class Set:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DataSet:
    
    def __init__(self, dataset, add_noise=None, noise_output=None):
        """
        Args:
            dataset: index of the dataset.
                1 - Letter recognition image data
                2 - Iris data
                3 - Glass identification data
                4 - Image segmentation data
                
                5 - Moons
                6 - Blobs
                7 - Circles
                8 - Gaussian Quantiles modified
                9 - Multiclass
                10 - Gaussian Quantiles
                11 - Noise
        """
        # load data
        x, y, self.class_names = get_data(dataset, binarize=True, preprocess=1)
        self.num_features = x.shape[1]
        self.num_classes = y.shape[1]
        #
        if add_noise == 2:
            x, y, self.class_names = add_noise_as_a_class(x, y,
                                                          self.class_names,
                                                          None, 100)
            self.num_classes += 1
        # split into 60%, 20% and 20%
        splt1 = train_test_split(x, y, test_size=0.6)
        splt2 = train_test_split(splt1[0], splt1[2], test_size=0.5)
        self.trn = Set(splt1[1], splt1[3])
        self.vld = Set(splt2[0], splt2[2])
        self.tst = Set(splt2[1], splt2[3])
        
        # add noise
        if add_noise == 1:
            self.trn.x, self.trn.y = add_noise_as_no_class(self.trn.x,
                                                           self.trn.y,
                                                           100, noise_output)