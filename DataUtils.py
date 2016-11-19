'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np
import random

def shuffle(a, b):
    if a.shape[0] != b.shape[0]: raise ValueError('sizes are different: a: ' + str(a.size) + ' and b: ' + str(b.size))
    
    ds = list(zip(a, b))
    random.shuffle(ds)
    ds = list(zip(*ds))
    
    return np.array(ds[0]), np.array(ds[1])

def split(ds_x, ds_y):
    if ds_x.shape[0] != ds_y.shape[0]: raise ValueError('sizes are different: a: ' + str(ds_x.size) + ' and b: ' + str(ds_y.size))
    a1 = []
    a2 = []
    b1 = []
    b2 = []
    for x, y in zip(ds_x, ds_y):
        if bool(random.getrandbits(1)):
            a1.append(x)
            b1.append(y)
        else:
            a2.append(x)
            b2.append(y)
    print("split %d into %d and %d" % (ds_x.shape[0], len(a1), len(a2)))
    return np.array(a1), np.array(b1), np.array(a2), np.array(b2)

"""
split into 50%, 25% and 25%
"""
def split_in_tvt(ds_x, ds_y):
    assert ds_x.shape[0] == ds_y.shape[0]
    d1 = count_distribution(ds_y)
    d2 = [0] * ds_y.shape[1]
    
    trn_x = []
    trn_y = []
    vld_x = []
    vld_y = []
    tst_x = []
    tst_y = []
    
    ds = list(zip(ds_x, ds_y))
    random.shuffle(ds)
    
    for x, y in ds:
        i = y.argmax()
        d2[i] += 1
        
        if d2[i]/d1[i] <= 0.5:
            trn_x.append(x)
            trn_y.append(y)
        elif d2[i]/d1[i] <= 0.75:
            vld_x.append(x)
            vld_y.append(y)
        else:
            tst_x.append(x)
            tst_y.append(y)
            
    count_distribution(np.array(trn_y))
    count_distribution(np.array(vld_y))
    count_distribution(np.array(tst_y))
    print("split %d into %d, %d and %d" % (ds_x.shape[0], len(trn_x), len(vld_x), len(tst_x)))
    return np.array(trn_x), np.array(trn_y), np.array(vld_x), np.array(vld_y), np.array(tst_x), np.array(tst_y)

"""
split into 60% and 40%
"""
def split_in_tt(ds_x, ds_y):
    assert ds_x.shape[0] == ds_y.shape[0]
    d1 = count_distribution(ds_y)
    d2 = [0] * ds_y.shape[1]
    
    trn_x = []
    trn_y = []
    tst_x = []
    tst_y = []
    
    ds = list(zip(ds_x, ds_y))
    random.shuffle(ds)
    
    for x, y in ds:
        i = y.argmax()
        d2[i] += 1
        
        if d2[i]/d1[i] <= 0.6:
            trn_x.append(x)
            trn_y.append(y)
        else:
            tst_x.append(x)
            tst_y.append(y)
            
    count_distribution(np.array(trn_y))
    count_distribution(np.array(tst_y))
    print("split %d into %d and %d" % (ds_x.shape[0], len(trn_x), len(tst_x)))
    return np.array(trn_x), np.array(trn_y), np.array(tst_x), np.array(tst_y)

def count_distribution(ds_y):
    d = [0] * ds_y.shape[1]
    
    for y in ds_y:
        d[y.argmax()] += 1
        
    #d = np.asarray(d) / ds_y.shape[0]
    print(d)
    return d

def remove_class(ds_x, ds_y, i):
    assert i >= 0 and i < ds_x.shape[1]
    
    new_ds_x = []
    new_ds_y = []
    outliers = []
    
    for x, y in zip(ds_x, ds_y):
        if y.argmax() == i:
            outliers.append(x)
        else:
            new_ds_x.append(x)
            new_ds_y.append(np.delete(y, i))
    
    return np.array(new_ds_x), np.array(new_ds_y), np.array(outliers)

def add_noise_as_no_class(ds_x, ds_y, noise_size=None, noise_output=None):
    assert ds_x.shape[0] == ds_y.shape[0]
    
    if noise_size is None:
        noise_size = ds_y.shape[0]
        
    if noise_output is None:
        noise_output = 1.0/ds_y.shape[1]
    
    noise = np.random.uniform(0.0, 1.0, [noise_size, ds_x.shape[1]])
    
    new_ds_x = np.concatenate([ds_x, noise])
    new_ds_y = np.concatenate([ds_y, np.array([[noise_output]*ds_y.shape[1]] * noise_size)])
    
    assert new_ds_x.shape[0] == new_ds_y.shape[0]
    return new_ds_x, new_ds_y

def add_noise_as_a_class(ds_x, ds_y, noise_size=None):
    assert ds_x.shape[0] == ds_y.shape[0]
    
    if noise_size is None:
        noise_size = ds_y.shape[0]
    
    noise = np.random.uniform(0.0, 1.0, [noise_size, ds_x.shape[1]])
    
    ds_y = np.append(ds_y, np.array([[0]] * ds_y.shape[0]), axis=1)
    
    new_ds_x = np.concatenate([ds_x, noise])
    new_ds_y = np.concatenate([ds_y, np.array([[0]*ds_y.shape[1] + [1]] * noise_size)])
    
    assert new_ds_x.shape[0] == new_ds_y.shape[0]
    return new_ds_x, new_ds_y