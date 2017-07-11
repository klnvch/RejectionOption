'''
Created on May 17, 2017

@author: anton

Copied from http://www.rueckstiess.net/research/snippets/show/72d2363e

TODO: impove performance of predict_proba
      init center properly

'''
from scipy import random, zeros, exp, dot
from scipy.linalg import norm, pinv
from scipy.spatial.distance import pdist
import numpy as np
from DataSet import DataSet
from sklearn.cluster import KMeans
from graphics import plot_decision_regions
from thresholds import thr_output
import time
from sklearn import metrics

class RBF:
    
    def __init__(self, n_features, n_centers, n_classes, beta=None):
        print('Create RBF network...')
        print('{:d}-{:d}-{:d}'.format(n_features, n_centers, n_classes))
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.beta = beta
    
    def _basisfunc(self, c, d):
        assert len(d) == self.n_features
        return exp(-self.beta * norm(c - d) ** 2)
    
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.n_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
    
    def set_random_centers(self, X):
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.n_centers]
        self.centers = [X[i, :] for i in rnd_idx]
        self.beta = 8.0
        print('center: {}'.format(self.centers))
    
    def set_kmeans_centers(self, X):
        # find centers
        kmeans = KMeans(n_clusters=self.n_centers)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        # find distance
        if self.beta is None:
            dists = pdist(self.centers)
            max_dist = dists.max()
            avg_dist = np.average(dists)
            self.beta = 8.0 / (2.0 * avg_dist ** 2)
            
            print('max distance: ' + str(max_dist) + 
                    ', avg distance: ' + str(avg_dist) + 
                    ', beta = ' + str(self.beta))
    
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
        #self.set_random_centers(X)
        self.set_kmeans_centers(X)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print(G)
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
    
    def predict_proba(self, X):
        if X is None: return None
        """ X: matrix of dimensions n x indim """
        #print('RBF.test: size {:d}'.format(len(X)))
        start_time = time.time()
        
        G = self._calcAct(X)
        Y = dot(G, self.W)
        
        print('RBF.test: {:9f} seconds'.format(time.time() - start_time))
        
        return Y
    
    def score(self, tst):
        outputs = self.predict_proba(tst.x)
        y_pred = outputs.argmax(axis=1)
        y_true = tst.y.argmax(axis=1)
        return metrics.accuracy_score(y_true, y_pred)

if __name__ == '__main__':
    ds = DataSet(12)
    rbf = RBF(ds.n_features, 16, ds.n_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, thr_output,
                          (4.1, 4.1), None, True, 0.05)