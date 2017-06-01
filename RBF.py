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
from data_utils import threshold_output

class RBF:
    
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for _ in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
    
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)
    
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
    
    def set_random_centers(self, X):
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]
        print('center: {}'.format(self.centers))
    
    def set_kmeans_centers(self, X):
        # find centers
        kmeans = KMeans(n_clusters=self.numCenters)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        # find distance
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
        # self.set_random_centers(X)
        self.set_kmeans_centers(X)
        # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)
        
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
    
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y
    
    def predict_proba(self, X):
        return self.test(X)

if __name__ == '__main__':
    ds = DataSet(5)
    rbf = RBF(ds.num_features, 16, ds.num_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, threshold_output,
                          (8, 8), None, 0.08)
