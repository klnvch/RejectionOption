'''
Created on Oct 26, 2016

@author: anton
'''

import numpy as np

def count_distribution(ds_y):
    d = [0] * ds_y.shape[1]
    
    for y in ds_y:
        d[y.argmax()] += 1
        
    #d = np.asarray(d) / ds_y.shape[0]
    print(d)
    return d

def remove_class(ds_x, ds_y, i):
    new_ds_x = []
    new_ds_y = []
    outliers = []
    
    for x, y in zip(ds_x, ds_y):
        if y.argmax() in i:
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
    
    noise = np.random.uniform(ds_x.min(), ds_x.max(), [noise_size, ds_x.shape[1]])
    
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