#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: General image plotting functions adapted from Geiss and Hardin 2021
"""

import bp_configs
import bp_utility
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LogNorm

FWID=7
fields = ['ref','t','q','u','v']
if bp_configs.USE_DOP_SPW:  
    fields = ['ref','vel','wid']
kazr_ranges = bp_configs.DATA_RANGE['kazr']
titles = {'ref': 'Reflectivity (dBZ)', 't': 'Temperature (K)', 'q': 'Specific Humidity (kg/kg)', 'u': 'u-Component Winds (m/$s^2$)', 'v': 'v-Component Winds (m/$s^2$)', 'vel': 'Doppler Velocity', 'wid': 'Spectral Width'}
kazr_res = [0.03, 0.078125] 
cmap = plt.cm.get_cmap('gist_ncar', 256)
cmap = cmap(np.linspace(0,1, 256))
cmap[:1, :] = np.array([0.8, 0.8, 0.8, 1])
custom_ref = colors.ListedColormap(cmap)
colormaps = {'ref': custom_ref, 't': 'Reds', 'q': 'Blues', 'u': 'Oranges', 'v': 'Greens', 'vel': 'bwr', 'wid': 'jet'}

def plot_data_field(data, field_name='ref'):
    data = np.double(data)
    data = bp_utility.inv_standardize(data,field_name,'kazr')
    
    dmn, dmx = kazr_ranges[field_name][0], kazr_ranges[field_name][1]
    colormap = colormaps[field_name]
    extent = (0,data.shape[1]*kazr_res[0], 0, data.shape[0]*kazr_res[1])
    aspect = kazr_res[0]/kazr_res[1]
    if field_name == 'q':
        im = plt.imshow(data,origin='lower',cmap=colormap, norm=LogNorm(vmin=dmn, vmax=dmx), extent=extent, aspect=aspect)
    else:
        im = plt.imshow(data,origin='lower',cmap=colormap,vmin=dmn, vmax=dmx, extent=extent, aspect=aspect)
    plt.colorbar(im)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Time (min)')
    plt.title(titles[field_name])

def plot_data(data, field_name='ref', fname=None):
    data = np.copy(data)
    if data.ndim == 2:
        f = plt.figure(figsize=[FWID, FWID*0.8])
        plot_data_field(data, field_name)
    elif data.ndim == 3:
        f = plt.figure(figsize=[FWID*bp_configs.CHANNELS, FWID*0.75])
        for i in range(bp_configs.CHANNELS):
            plt.subplot(1,bp_configs.CHANNELS,i+1)
            plot_data_field(data[:,:,i], fields[i])
    elif data.ndim == 5:
        f = plt.figure(figsize=[FWID*bp_configs.CHANNELS, FWID*0.66])
        for i in range(bp_configs.CHANNELS):
            plt.subplot(1,bp_configs.CHANNELS,i+1)
            plot_data_field(data[:,:,i], fields[i])
    if fname is not None:
        plt.savefig(fname)
        plt.close(f)

def shap_plot(data,field_name='ref',fname=None):
    data = np.copy(data)
    f = plt.figure(figsize=[FWID, FWID*0.8])
    plot_data_field(data[:,:,0], field_name)
        
def show():
    plt.show()

def plot(data, case, field_name='ref', fname=None):
    plot_data(data, field_name, fname)