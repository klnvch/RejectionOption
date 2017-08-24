'''
Created on Oct 29, 2016

@author: anton
'''
import matplotlib.pyplot as plt

def plot_2d_dataset(x, y, figsize=(4.1, 4.1), savefig=None):
    if y.ndim == 2: y = y.argmax(axis=1)
    
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.scatter(x[:, 0], x[:, 1], c=y)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()