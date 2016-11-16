'''
Created on Oct 29, 2016

@author: anton
'''

import numpy as np
import matplotlib.pyplot as plt

def draw(c, e, rc, re, axis=1):
    
    if c.shape[0] == 0: c = np.array([[None, None]])
    if e.shape[0] == 0: e = np.array([[None, None]])
    if rc.shape[0] == 0: rc = np.array([[None, None]])
    if re.shape[0] == 0: re = np.array([[None, None]])
    
    lbl_c,  = plt.plot(*zip(*c),  marker='s', color='green', markersize='5', ls='', label='Correct')
    lbl_e,  = plt.plot(*zip(*e),  marker='p', color='red',   markersize='5', ls='', label='Errors')
    lbl_rc, = plt.plot(*zip(*rc), marker='o', color='blue',  markersize='5', ls='', label='Rejected correct')
    lbl_re, = plt.plot(*zip(*re), marker='o', color='cyan',  markersize='5', ls='', label='Rejected errors')
    
    plt.legend(handles=[lbl_c, lbl_e, lbl_rc, lbl_re], numpoints=1, loc=2)
    plt.axis([-axis, axis, -axis, axis])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()


def draw_x_vs_y(xs, ys, xlabel=None, ylabel=None, labels=None, colors=None, legend_location=2):
    handles = []
    
    for x, y, label, color in zip(xs, ys, labels, colors):
        lbl, = plt.plot(x, y, color=color, linewidth=2.0, label=label)
        handles.append(lbl)
    
    plt.legend(handles=handles, numpoints=1, loc=legend_location)
    if xlabel is not None and ylabel is not None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.show()