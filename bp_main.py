#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: The main runscript for the project used for training models
"""

import os, gc
import numpy as np
import bp_utility
import bp_plotting
import bp_batch
import bp_models
import bp_configs
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.callbacks import History
from keras.optimizers import adam_v2
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D, Dense
from keras.layers import LeakyReLU, Lambda, concatenate, Dropout, Flatten
from keras.models import Model
from scipy.signal import convolve2d as conv
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

if tf.test.gpu_device_name(): 
    print('Default GPU Device {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

######################## DATA LOADER START
###############################################################################################################

def data_combiner(paths, target, axis):
    combined_arr = []
    sample_indices = -1
    test_breaks = []
    test_details = []
    offset = -1
    for i, path in enumerate(paths):
        print(bp_configs.data_dir + target + '/' + target + '_' + path + '_kazr.npy')
        if target == "preprocessed" and os.path.isfile(bp_configs.data_dir + target + '/' + path + '_kazr.npy'):
            temp_data = np.load(bp_configs.data_dir + target + '/' + path + '_kazr.npy')
        elif os.path.isfile(bp_configs.data_dir + target + '/' + target + '_' + path + '_kazr.npy'):
                temp_data = np.load(bp_configs.data_dir + target + '/' + target + '_' + path + '_kazr.npy')
        else:
            continue

        if target == "test_set":
            test_breaks.append(temp_data.shape[0])
            test_details.append(path)

        print(path, temp_data.shape)
        if len(combined_arr) == 0: #basecase
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

# Build paths
data_paths = bp_utility.path_builder()
# print("Loading data from:", data_paths)

comb_test_set = -1
test_breaks = -1

# Load data
# Preprocessed
comb_preprocessed, _, _ = data_combiner(data_paths, 'preprocessed', axis=1)
print("\nPreprocessed loaded", comb_preprocessed.shape)
print("Total Size of Training set = ", comb_preprocessed.shape[1]/128)

# Indices
comb_indices, DOWNFILL_SAMPLE_INDS, _ = data_combiner(data_paths, 'indices', axis=0)
print("\nIndices loaded", comb_indices.shape)

# Samples
comb_samples, _, _ = data_combiner(data_paths, 'samples', axis=0)
print("\nSamples loaded", comb_samples.shape)

# Test Sets
# comb_test_set, _, test_breaks = data_combiner(data_paths, 'test_set', axis=0)
# print("\nTest sets loaded", comb_test_set.shape)
# # print(test_breaks)

print("All data loaded!")


########### DEFINE PLOTTING FUNCTIONS  ########################################

# Train the unetpp from Geiss and Hardin, 2021
def train_l1(case):
    LR_SCHEDULE = [450,475]
    base_channels=8
    levels=7
        
    # Start a run
    wandb.init(
        # set the wandb project where this run will be logged
        project=bp_configs.RUN_NAME,
    )

    history = History()
    config_defaults = bp_configs.config_defaults
    cnn = bp_models.unetpp((*bp_configs.SIZE[case],bp_configs.CHANNELS+1),base_channels=base_channels,levels=levels,growth=2)
    cnn.summary()
    cnn.compile(optimizer=adam_v2.Adam(learning_rate=config_defaults['lr']), loss=bp_models.blind_MAE)
    batch = bp_batch.BATCH_FUNC[case]

    data = comb_preprocessed[:,:,:bp_configs.CHANNELS]
    x_samples = comb_samples[:,:,:,:bp_configs.CHANNELS+1]
    if bp_configs.CHANNELS == 1:
        x_samples = np.squeeze(tf.stack([comb_samples[:,:,:,:bp_configs.CHANNELS], comb_samples[:,:,:,-bp_configs.CHANNELS:]], axis=3))
    x = np.zeros((250*config_defaults['batch_size'],*bp_configs.SIZE[case],bp_configs.CHANNELS+1),dtype='float16')
    
    #the training loop:
    for epoch in range(config_defaults['epochs']):
        gc.collect()
        print('EPOCH ' + str(epoch) + ':')
        
        if epoch in LR_SCHEDULE:
            print('Reducing Learning Rate...')
            K.set_value(cnn.optimizer.lr,K.get_value(cnn.optimizer.lr)*0.1)

        batch(x, data, sample_inds = DOWNFILL_SAMPLE_INDS)
        print("L1: Epoch batch size", x.shape)
        cnn.fit(x,x,batch_size=config_defaults['batch_size'],verbose=1, validation_split=0.05, callbacks=[history, WandbMetricsLogger(log_freq=5)])
        np.save(bp_configs.project_path + 'models/' + case + '_l1/' + bp_configs.RUN_CASE + 'loss.npy',np.array(history.history['loss']))
        
        if epoch%50 == 0:
            cnn.save(bp_configs.project_path + 'models/' + case + '_l1/' + bp_configs.RUN_CASE + 'epoch_' + str(epoch+1).zfill(4))
            
            outputs = cnn.predict(x_samples,batch_size=config_defaults['batch_size'])
            if bp_configs.CHANNELS == 3:
                r,v,w = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2]
                v[r<-0.5] = -1.0
                w[r<-0.5] = -1.0
                r[r<-0.5] = -1.0 
                outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2] = r, v, w
            elif bp_configs.CHANNELS == 5:
                r,v,w,uw,vw = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2], outputs[:,:,:,3], outputs[:,:,:,4]
                v[r<-0.5] = -1.0
                w[r<-0.5] = -1.0
                r[r<-0.5] = -1.0 
                uw[r<-0.5] = -1.0 
                vw[r<-0.5] = -1.0 
                outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2], outputs[:,:,:,3], outputs[:,:,:,4] = r, v, w, uw, vw
            else:
                r = outputs[:,:,:,0]
                r[r<-0.5] = -1.0
                outputs[:,:,:,0] = r
            for i in range(outputs.shape[0]):
                bp_plotting.plot(outputs[i,:,:,:bp_configs.CHANNELS],case,fname=bp_configs.image_path + case + '_l1/' + bp_configs.RUN_CASE + '/sample' + str(i) + '_epoch' + str(epoch+1).zfill(4) + '.png')
            gc.collect()

    wandb.finish()
            
# Train the 3Net+
def train_3net(case):
    LR_SCHEDULE = [450,475]

    # Start a run
    wandb.init(
        # set the wandb project where this run will be logged
        project=bp_configs.RUN_NAME,
    )

    history = History()
    config_defaults = bp_configs.config_defaults

    cnn = bp_models.unet3plus((*bp_configs.SIZE[case],bp_configs.CHANNELS+1), bp_configs.CHANNELS, \
                               config=config_defaults, depth=config_defaults['depth'], training=False, clm=False)
    cnn.summary()
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config_defaults['lr']), loss=bp_models.blind_MAE, metrics=[bp_models.dice_coef])
    batch = bp_batch.BATCH_FUNC[case]
    
    data = comb_preprocessed[:,:,:bp_configs.CHANNELS]
    x_samples = comb_samples[:,:,:,:bp_configs.CHANNELS+1]
    # x_samples = np.concatenate([comb_samples[:,:,:,:bp_configs.CHANNELS], np.expand_dims(comb_samples[:,:,:,bp_configs.CHANNELS+1], axis=-1)], axis=3)

    if bp_configs.CHANNELS == 1:
        x_samples = np.squeeze(tf.stack([comb_samples[:,:,:,:bp_configs.CHANNELS], comb_samples[:,:,:,-bp_configs.CHANNELS:]], axis=3))
    
    x = np.zeros((250*config_defaults['batch_size'],*bp_configs.SIZE[case],bp_configs.CHANNELS+1),dtype='float16')
    
    for epoch in range(config_defaults['epochs']):
        gc.collect()
        print('EPOCH ' + str(epoch) + ':')
        
        if epoch in LR_SCHEDULE:
            print('Reducing Learning Rate...')
            K.set_value(cnn.optimizer.lr,K.get_value(cnn.optimizer.lr)*0.1)

        batch(x, data, sample_inds = DOWNFILL_SAMPLE_INDS)
        print("L1: Epoch batch size", x.shape)

        cnn.fit(x, x, batch_size=config_defaults['batch_size'], verbose=1, validation_split=0.05, callbacks=[history, WandbMetricsLogger(log_freq=5)])
        np.save(bp_configs.project_path + 'models/' + case + '_3net/' + bp_configs.RUN_CASE + 'loss.npy', np.array(history.history['loss']))
        
        if epoch%50 == 0:
            cnn.save(bp_configs.project_path + 'models/' + case + '_3net/' + bp_configs.RUN_CASE + '/epoch_' + str(epoch+1).zfill(4))
            
            outputs = cnn.predict(x_samples,batch_size=config_defaults['batch_size'])#[0]
            print(outputs.shape)
            if bp_configs.CHANNELS == 3:
                r,v,w = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2]
                v[r<-0.5] = -1.0
                w[r<-0.5] = -1.0
                r[r<-0.5] = -1.0 
                outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2] = r, v, w
            elif bp_configs.CHANNELS == 4:
                r,v,w,uw = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2], outputs[:,:,:,3]
                v[r<-0.5] = -1.0
                w[r<-0.5] = -1.0
                r[r<-0.5] = -1.0 
                uw[r<-0.5] = -1.0 
                outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2], outputs[:,:,:,3]= r, v, w, uw
            elif bp_configs.CHANNELS == 5:
                r,v,w,uw,vw = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2], outputs[:,:,:,3], outputs[:,:,:,4]
                v[r<-0.5] = -1.0
                w[r<-0.5] = -1.0
                r[r<-0.5] = -1.0 
                uw[r<-0.5] = -1.0 
                vw[r<-0.5] = -1.0 
                outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2], outputs[:,:,:,3], outputs[:,:,:,4] = r, v, w, uw, vw
            else:
                r = outputs[:,:,:,0]
                r[r<-0.5] = -1.0
                outputs[:,:,:,0] = r
            for i in range(outputs.shape[0]):
                bp_plotting.plot(outputs[i,:,:,:bp_configs.CHANNELS],case,fname=bp_configs.image_path + case + '_3net/' + bp_configs.RUN_CASE + '/sample' + str(i) + '_epoch' + str(epoch+1).zfill(4) + '.png')
            gc.collect()

    wandb.finish()

def print_samples(case):
    x_samples = comb_samples
    for i,x in enumerate(x_samples):
        print(x.shape)
        bp_plotting.plot(x[:,:,:3], case, fname=bp_configs.project_path + 'figures/training_samples/base/base_' + str(i) + '.png')
                
        fig = plt.figure(figsize=(10,10))
        plt.imshow(np.double(x[:,:,3]), cmap="bwr")
        plt.title("Sample " + str(i) + "Mask")
        plt.gca().invert_yaxis()
        plt.savefig(bp_configs.project_path + 'figures/training_samples/base/base_mask_' + str(i) + '.png')

######## MAIN RUNLOOP
if __name__ == '__main__':

    # Print models
    # from keras.utils import plot_model
    # cnn = unetpp((256,256,4),base_channels=8,levels=7,growth=2)
    # plot_model(cnn,'./figures/cnn_plots/l1_downfill.png',show_shapes=True)

    print("Training models..")

    print("Training unet3+...")
    train_3net('downfill')

    # print("Training l1...")
    # train_l1('downfill')

    # print("Training cgan")
    # train_cgan('downfill')

    # print("\nPrint samples")
    # print_samples('downfill')

    print("All done!")