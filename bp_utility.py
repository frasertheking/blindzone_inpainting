#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Adapted from Geiss and Hardin, 2021. This file holds general utility files for the project including standardization
"""

import os
import bp_configs
import numpy as np

def path_builder():
    data_paths = []
    for site in bp_configs.SITES_TO_EXAMINE:
        for year in bp_configs.YEARS_TO_EXAMINE:
            for month in bp_configs.MONTHS_TO_EXAMINE:
                data_paths.append(site + '_' + year + '_' + month)
    return data_paths

def data_combiner(paths, prefix, target, axis):
    combined_arr = []
    sample_indices = -1
    test_breaks = []
    test_details = []
    offset = -1
    for i, path in enumerate(paths):
        print("Loading: " + prefix + target + '/' + target + '_' + path + '_kazr.npy')
        if os.path.isfile(prefix + target + '/' + target + '_' + path + '_kazr.npy'):
                temp_data = np.load(prefix + target + '/' + target + '_' + path + '_kazr.npy')
        else:
            continue

        if target == "test_set":
            test_breaks.append(temp_data.shape[0])
            test_details.append(path)

        print(path, temp_data.shape)
        if len(combined_arr) == 0: # basecase
            combined_arr = temp_data
            if target == "indices":
                sample_indices = combined_arr[:int(len(combined_arr)*(1-bp_configs.TEST_FRAC))]
                offset = temp_data.shape[0]
        else:
            if target == "indices":
                combined_arr = np.concatenate([combined_arr, (temp_data + offset)], axis=axis)
                sample_indices = np.concatenate([sample_indices, (temp_data + offset)[:int(len((temp_data + offset))*(1-bp_configs.TEST_FRAC))]], axis=0)
                offset = offset = temp_data.shape[0]
                print("offset", offset)
            else:
                combined_arr = np.concatenate([combined_arr, temp_data], axis=axis)

    return combined_arr, sample_indices, test_breaks

def standardize(x, field, instrument):
    mn,mx = bp_configs.DATA_RANGE[instrument][field]
    if field == 'ref':
        x[np.abs(x)>100] = -60.0
        x = 1.5*(x-mn)/(mx-mn)-0.5
        x[x<=-0.5] = -1.0 
        x[x>1.0] = 1.0
    elif field == 't':
        x = 1.5*(x-mn)/(mx-mn)-0.5
    elif field == 'q':
        x = 1.5*(x-mn)/(mx-mn)-0.5
    elif field == 'u':
        x = 1.5*(x-mn)/(mx-mn)-0.5
    elif field == 'v':
        x = 1.5*(x-mn)/(mx-mn)-0.5
    elif field == 'vel':
        x[np.abs(x)>100] = 0.0
        x = np.tanh(1.5*x/mx)
    elif field == 'wid':
        x[np.abs(x)>100] = 0.0
        x = 2.0*x/mx-1.0
        x[x>1.0] = 1.0
        x[x<-1.0] = -1.0
    return x

def inv_standardize(x,field,instrument):
    mn,mx = bp_configs.DATA_RANGE[instrument][field]
    if field == 'ref':
        x[x<-0.5] = -0.5
        x = (x+0.5)*(mx-mn)/1.5+mn
    elif field == 't':
        x = (x+0.5)*(mx-mn)/1.5+mn
    elif field == 'q':
        x = (x+0.5)*(mx-mn)/1.5+mn
    elif field == 'u':
        x = (x+0.5)*(mx-mn)/1.5+mn
    elif field == 'v':
        x = (x+0.5)*(mx-mn)/1.5+mn
    elif field == 'vel':
        x = np.arctanh(x)*mx/1.5
    elif field == 'wid':
        x = mx*(x+1.0)/2.0
    return x
    
def boxcar1d(x,N):
    filtered = np.zeros((x.shape[0]-N+1, x.shape[1]), x.dtype)
    mn = np.sum(x[:N,:], axis=0)
    for i in range(x.shape[0]-N):
        filtered[i,:] = mn/N
        mn += x[i+N,:] - x[i,:]
    filtered[i+1,:] = mn/N
    return filtered

def boxcar2d(x, N):
    x = boxcar1d(x, N)
    x = boxcar1d(x.T, N).T
    return x
