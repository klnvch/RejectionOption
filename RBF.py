'''
Created on May 17, 2017

@author: anton

Copied from http://www.rueckstiess.net/research/snippets/show/72d2363e

TODO: impove performance of predict_proba
      init center properly

'''
from scipy import random, exp, dot
from scipy.linalg import norm, pinv
import math
import numpy as np
from DataSet import DataSet
from sklearn.cluster import KMeans
from graphics import plot_decision_regions
import time
from sklearn import metrics
from klnvch.rejection_option.thresholds import Thresholds

def get_kmeans_centers(X, k):
    # find centers
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    variances = np.zeros(k)
    # find distance
    for i in range(k):
        dists = [norm(centers[i] - c) for c in centers]
        dists = np.sort(dists)
        variances[i] = (dists[1] + dists[2] + dists[3]) / 3.0
    
    print(variances)
    return centers, variances

class RBF:
    
    def __init__(self, n_features, n_centers, n_classes, beta=None):
        print('Create RBF network...')
        print('{:d}-{:d}-{:d}'.format(n_features, n_centers, n_classes))
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.beta = np.full(n_centers, beta)
    
    def _basisfunc(self, i, x):
        assert len(x) == self.n_features
        return exp(self.beta[i] * norm(self.centers[i] - x) ** 2)
    
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    def _calcAct(self, X):
        # calculate activations of RBFs
        # init matrix with extra column for bias
        G = np.ones((X.shape[0], self.n_centers + 1))
        for i in range(self.n_centers):
            for j, x in enumerate(X):
                G[j, i] = self._basisfunc(i, x)
        return G
    
    def set_random_centers(self, X):
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.n_centers]
        self.centers = [X[i, :] for i in rnd_idx]
        self.beta = np.full(self.n_centers, 8.0)
        print('center: {}'.format(self.centers))
    
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
        #self.set_random_centers(X)
        self.centers, variances = get_kmeans_centers(X, self.n_centers)
        self.beta = - 1.0 / (2.0 * variances**2)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print(G)
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
    
    def predict_proba(self, X):
        if X is None: return None
        X = np.array(X)
        """ X: matrix of dimensions n x indim """
        #print('RBF.test: size {:d}'.format(len(X)))
        start_time = time.time()
        
        G = self._calcAct(X)
        Y = dot(G, self.W)
        
        sigmoid = np.vectorize(self.sigmoid)
        Y = sigmoid(Y)
        
        print('RBF.test: {:9f} seconds'.format(time.time() - start_time))
        
        return Y
    
    def score(self, tst):
        outputs = self.predict_proba(tst.x)
        y_pred = outputs.argmax(axis=1)
        y_true = tst.y.argmax(axis=1)
        return metrics.accuracy_score(y_true, y_pred)

if __name__ == '__main__':
    ds = DataSet(9)
    rbf = RBF(ds.n_features, 20, ds.n_classes)
    rbf.train(ds.trn.x, ds.trn.y)
    plot_decision_regions(ds.tst.x, ds.tst.y, rbf, Thresholds.thr_output,
                          (4.1, 4.1), None, True, 0.05)