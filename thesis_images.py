'''
Created on May 23, 2017

@author: anton

This module keeps function to generate images for the thesis text

Those images are not part of results and are only used to illustrate 
some basic definitions 
'''

import matplotlib.pyplot as plt
from graphics import get_cmap
from input_data import get_data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


FIG_HALF_SIZE = 4.1
    

def plot_pca_vs_lda(X, y, target_names):
    """
    Chapter Rejection option. Section Sources of errors
    
    Copied from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
    """
    n_classes = len(target_names)
    
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

    fig = plt.figure(figsize=(FIG_HALF_SIZE, FIG_HALF_SIZE))
    fig.canvas.set_window_title('PCA') 
    colors = get_cmap(n_classes)
    lw = 2

    for i in range(n_classes):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=colors(i), alpha=.8, lw=lw, s=4,
                label=target_names[i])
    plt.legend(loc='best', shadow=False, scatterpoints=1, markerscale=4)
    plt.savefig('/home/anton/Desktop/diploma_text/images/chapter_3/pca.png')
    #plt.title('PCA')

    fig = plt.figure(figsize=(FIG_HALF_SIZE, FIG_HALF_SIZE))
    fig.canvas.set_window_title('LDA') 
    for i in range(n_classes):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=colors(i), s=4,
                label=target_names[i])
    plt.legend(loc='best', shadow=False, scatterpoints=1, markerscale=4)
    plt.savefig('/home/anton/Desktop/diploma_text/images/chapter_3/lda.png')
    #plt.title('LDA')

    plt.show()
    
    
def plot_roc_space_figure_1():
    """
    ROC space with five discrete rejection
    """
    plt.figure(figsize=(FIG_HALF_SIZE, FIG_HALF_SIZE))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot([.1, .4, .7, .0, .6], [.6, .8, .7, 1., .2], 'ro')
    plt.annotate('A', xy=(.12, .55))
    plt.annotate('B', xy=(.42, .75))
    plt.annotate('C', xy=(.72, .65))
    plt.annotate('D', xy=(.02, .95))
    plt.annotate('E', xy=(.62, .15))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('/home/anton/Desktop/diploma_text/images/roc_space.png')
    plt.show()
    
    
    

def plot_roc_space_figure_2():
    """
    ROC space with three ROC curves
    """
    plt.figure(figsize=(FIG_HALF_SIZE, FIG_HALF_SIZE))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.96, 0.97, 0.98, 0.99, 1], 'r')
    plt.annotate('A', xy=(.05, .7))
    plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.93, 0.95, 0.97, 1], 'b')
    plt.annotate('B', xy=(.25, .65))
    plt.plot([0, 0.4, 1], [0, 0.8, 1], 'g')
    plt.annotate('C', xy=(.3, .55))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('/home/anton/Desktop/diploma_text/images/roc_curves.png')
    plt.show()


if __name__ == '__main__':
    # Sources of errors
    x, y, classes = get_data(4, preprocess=None)
    plot_pca_vs_lda(x, y, classes)
    
    #plot_roc_space_figure_1()
    #plot_roc_space_figure_2()