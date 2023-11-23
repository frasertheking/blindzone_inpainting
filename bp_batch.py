
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Input minibatch selection code used during model training adapted from Geiss and Hardin, 2021
"""

import numpy as np
import bp_configs
import bp_utility

DOWNFILL_SAMPLE_INDS = -1

def downfill_batch(x,data = None, sample_inds=DOWNFILL_SAMPLE_INDS, cut_range=bp_configs.DOWNFILL_CUT_RANGE, buf_range=bp_configs.BUF_RANGE):
    BS = x.shape[0]
    NT = x.shape[2]
    
    if not data is None:
        for i in range(BS):
            idx = np.random.choice(sample_inds)
            x[i,:,:,:bp_configs.CHANNELS]  = np.copy(data[:,idx:idx+NT,:])
            if np.random.choice([True, False]):
                x[i,:,:,:] = np.flip(x[i,:,:,:],axis=1)
        
    FS = bp_configs.min_weather_size
    for i in range(BS):
        if len(cut_range)==2:
            mask = np.float16(x[i,cut_range[0]:cut_range[1]+FS,:,0]>-bp_configs.CLOUD_MASK)
            mask = bp_utility.boxcar2d(mask,FS)>0.99
            valid_levs = np.where(np.any(mask,axis=1))[0]
            if len(valid_levs)>0:
                cut_ind = np.random.choice(valid_levs)+cut_range[0]
            else:
                cut_ind = np.random.randint(cut_range[0],cut_range[1])
        else:
            cut_ind = cut_range[0]
        N_buf = np.random.randint(buf_range[0],buf_range[1])
        buf = np.linspace(1.0,0.0,N_buf+2)[1:-1]
        mask = np.zeros((x.shape[1],x.shape[2]),dtype='float16')
        mask[:cut_ind,:] = 1.0
        mask[cut_ind:cut_ind+N_buf,:] = buf[:,np.newaxis]
        x[i,:,:,-1] = mask

BATCH_FUNC = {'downfill': downfill_batch}