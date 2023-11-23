#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Adapted from Geiss and Hardin 2021, we define the various linear inpainting techniques here
"""

import numpy as np

def find_edge_pixels(mask):
    mask = np.pad(mask,pad_width=1, mode='reflect')
    edge = np.logical_and(mask[1:-1,1:-1], (mask[:-2,1:-1]+mask[2:,1:-1]+mask[1:-1,2:]+mask[1:-1,:-2])<4)
    return np.where(edge)

def marching_avg(scan, mask, WIN_SIZE=8):
    while np.any(mask):
        IE, JE = find_edge_pixels(mask)
        pad_scan = np.pad(scan, WIN_SIZE, 'reflect')
        pad_mask = 1.0-np.pad(mask, WIN_SIZE, 'reflect')
        W = WIN_SIZE*2+1
        for ie, je in zip(IE,JE):
            scan_region = pad_scan[ie:ie+W,je:je+W]
            mask_region = pad_mask[ie:ie+W,je:je+W]
            masked_conv = np.sum(scan_region*mask_region)/np.sum(mask_region)
            scan[ie,je] = masked_conv
            mask[ie,je] = 0.0
    return scan

def repeat(scan, mask):
    BH = np.where(mask[:,0]==0)[0][0]
    sample = scan[BH,:]
    for i in range(BH-1,-1,-1):
        scan[i,:] = sample
    return scan