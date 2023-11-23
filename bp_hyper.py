#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Hyperparameter sweep code using wandb
"""

import os, gc
import bp_configs
import bp_batch
import bp_utility
import bp_loss
import wandb
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import keras.backend as Kb
from wandb.keras import WandbMetricsLogger
from keras.layers import Lambda, concatenate
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'epoch/val_loss'
    },
    'parameters': {
        'depth': {
            'values': [3, 4, 5, 6]
        },
        'batch_size': {
            'values': [2, 4, 8, 16]
        },
        'epochs': {
            'values': [8, 16, 32]
        },
        'lr': {
            'values': [5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]
        },
        'l2_reg': {
            'values': [1e-2, 1e-3, 1e-4, 1e-5]
        },
        'filters': {
            'values': [16, 32, 64]
        },
        'dropout': {
            'values': [0.01, 0.1, 0.2, 0.5]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'interpolation': {
            'values': ['bilinear', 'nearest']
        }
     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='3net-full-sweep')

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

comb_preprocessed, _, _ = data_combiner(data_paths, 'preprocessed', axis=1)
print("\nPreprocessed loaded", comb_preprocessed.shape)
print("Total Size of Training set = ", comb_preprocessed.shape[1]/128)

# Indices
comb_indices, DOWNFILL_SAMPLE_INDS, _ = data_combiner(data_paths, 'indices', axis=0)
print("\nIndices loaded", comb_indices.shape)

# We need to redefine a few helper functions here for this sweep
def blind_MAE(y_true,y_pred):
    mae = tf.math.abs(y_true[:,:,:,:bp_configs.CHANNELS]-y_pred)
    filt = y_true[:,:,:,bp_configs.CHANNELS]
    if bp_configs.CHANNELS == 3:
        filt = tf.stack([filt,filt,filt],axis=3)
    else:
        filt = tf.stack([filt],axis=3)
    
    weighted_mae = tf.reduce_mean(mae*filt)/tf.reduce_mean(filt)
    return weighted_mae

def preprocess(x):
    filt = tf.floor(x[:,:,:,bp_configs.CHANNELS])
    x0 = x[:,:,:,0]*(1-filt) - filt    #set missing ref to -1
    if bp_configs.CHANNELS == 3:
        if bp_configs.USE_DOP_SPW:
            x1 = x[:,:,:,1]*(1-filt)           #set missing vel to 0
            x2 = x[:,:,:,2]*(1-filt) - filt    #set missing wid to -1
        else:
            x1 = x[:,:,:,1]*(1-filt) - filt          #set missing vel to 0
            x2 = x[:,:,:,2]*(1-filt) - filt    #set missing wid to -1
        x = tf.stack([x0,x1,x2,x[:,:,:,3]],axis=3)
        return Lambda(lambda x: x)(x)
    else:
        x = tf.stack([x0,x[:,:,:,bp_configs.CHANNELS]],axis=3)
        return Lambda(lambda x: x)(x)  

def merge_output(xin,x):
    filt = xin[:,:,:,bp_configs.CHANNELS]
    nchan = bp_configs.CHANNELS
    outputs = []
    for i in range(nchan):
        merged = filt*x[:,:,:,i] + (1.0-filt)*xin[:,:,:,i]
        outputs.append(Kb.expand_dims(merged))
    return Lambda(lambda x: x)(concatenate(outputs))


"""
Custom UNet3+ with Deep Supervision & Dropout
"""
def conv_block(x, kernels, kernel_size=(3, 3), strides=(1, 1), padding='same', is_bn=True, is_relu=True, n=2, l2_reg=1e-4):
    for _ in range(1, n+1):
        x = k.layers.Conv2D(filters=kernels, kernel_size=kernel_size,
                            padding=padding, strides=strides,
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                            kernel_initializer=k.initializers.he_normal(seed=42))(x)
        if is_bn:
            x = k.layers.BatchNormalization()(x)
        if is_relu:
            x = k.activations.relu(x)
    return x

def unet3plus_deepsup(input_shape, output_channels, config, depth=4, training=False):
    """ Prep """
    interp = config.interpolation
    input_layer = k.layers.Input(shape=input_shape, name="input_layer")
    xpre = preprocess(input_layer)

    """ Encoder """
    encoders = []
    for i in range(depth+1):
        if i == 0:
            e = conv_block(xpre, config.filters*(2**i), l2_reg=config.l2_reg)
        else:
            e = k.layers.MaxPool2D(pool_size=(2, 2))(encoders[i-1])
            e = k.layers.Dropout(config.dropout)(e)
            e = conv_block(e, config.filters*(2**i), l2_reg=config.l2_reg)
        encoders.append(e)

    """ Middle """
    cat_channels = config.filters
    cat_blocks = depth+1
    upsample_channels = cat_blocks * cat_channels

    """ Decoder """
    decoders = []
    for d in reversed(range(depth+1)):
        if d == 0 :
            continue
        loc_dec = []
        decoder_pos = len(decoders)
        for e in range(len(encoders)):
            if d > e+1:
                e_d = k.layers.MaxPool2D(pool_size=(2**(d-e-1), 2**(d-e-1)))(encoders[e])
                e_d = k.layers.Dropout(config.dropout)(e_d)
                e_d = conv_block(e_d, cat_channels, n=1, l2_reg=config.l2_reg)
            elif d == e+1:
                e_d = conv_block(encoders[e], cat_channels, n=1, l2_reg=config.l2_reg)
            elif e+1 == len(encoders):
                e_d = k.layers.UpSampling2D(size=(2**(e+1-d), 2**(e+1-d)), interpolation=interp)(encoders[e])
                e_d = k.layers.Dropout(config.dropout)(e_d)
                e_d = conv_block(e_d, cat_channels, n=1, l2_reg=config.l2_reg)
            else:
                e_d = k.layers.UpSampling2D(size=(2**(e+1-d), 2**(e+1-d)), interpolation=interp)(decoders[decoder_pos-1])
                e_d = k.layers.Dropout(config.dropout)(e_d)
                e_d = conv_block(e_d, cat_channels, n=1, l2_reg=config.l2_reg)
                decoder_pos -= 1
            loc_dec.append(e_d)
        de = k.layers.concatenate(loc_dec)
        de = conv_block(de, upsample_channels, n=1, l2_reg=config.l2_reg)
        decoders.append(de)

    """ Final """
    d1 = decoders[len(decoders)-1]
    d1 = conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False, l2_reg=config.l2_reg)
    d1 = k.activations.tanh(d1)
    outputs = [merge_output(input_layer, d1)]

    """ Deep Supervision """
    if training:
        for i in reversed(range(len(decoders))):
            if i == 0:
                e = conv_block(encoders[len(encoders)-1], output_channels, n=1, is_bn=False, is_relu=False, l2_reg=config.l2_reg)
                e = k.layers.UpSampling2D(size=(2**(len(decoders)-i), 2**(len(decoders)-i)), interpolation=interp)(e)
                e = k.layers.Dropout(config.dropout)(e)
                e = k.activations.tanh(e)
                outputs.append(merge_output(input_layer, e))
            else:
                d = conv_block(decoders[i - 1], output_channels, n=1, is_bn=False, is_relu=False, l2_reg=config.l2_reg)
                d = k.layers.UpSampling2D(size=(2**(len(decoders)-i), 2**(len(decoders)-i)), interpolation=interp)(d)
                e = k.layers.Dropout(config.dropout)(e)
                d = k.activations.tanh(d)
                outputs.append(merge_output(input_layer, d))

    return tf.keras.Model(inputs=input_layer, outputs=outputs, name='UNet3Plus_DeepSup')
    
# The sweep calls this function with each set of hyperparameters
def train():
    config_defaults = {
        'depth': 4,
        'batch_size': 8,
        'epochs': 16,
        'lr': 5e-4,
        'l2_reg': 1e-4,
        'filters': 64,
        'dropout': 0.1,
        'optimizer': 'adam',
        'interpolation': 'bilinear',
        'momentum': 0.9,
        'seed': 42
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    EPOCH_SIZE = 250*config.batch_size
    cnn = unet3plus_deepsup((*bp_configs.SIZE['downfill'], bp_configs.CHANNELS+1), 3, config=config, depth=config.depth, training=True)
    cnn.summary()

    # Define the optimizer
    optimizer = -1
    if config.optimizer=='sgd':
      optimizer = SGD(learning_rate=config.lr, momentum=config.momentum, nesterov=True)
    elif config.optimizer=='rmsprop':
      optimizer = RMSprop(learning_rate=config.lr)
    elif config.optimizer=='adam':
      optimizer = Adam(learning_rate=config.lr)

    cnn.compile(optimizer=optimizer, loss=bp_loss.unet3p_hybrid_loss)
    batch = bp_batch.BATCH_FUNC['downfill']

    data = comb_preprocessed[:,:,:bp_configs.CHANNELS]
    x = np.zeros((EPOCH_SIZE, *bp_configs.SIZE['downfill'], bp_configs.CHANNELS+1), dtype='float16')

    for epoch in range(config.epochs):
        gc.collect()
        print('EPOCH ' + str(epoch) + ':')
        batch(x, data, sample_inds = DOWNFILL_SAMPLE_INDS)
        cnn.fit(x, x, batch_size=config.batch_size, verbose=1, validation_split=0.05, callbacks=[WandbMetricsLogger(log_freq=5), EarlyStopping(patience=10, restore_best_weights=True)])

######## MAIN RUNLOOP
if __name__ == '__main__':
    print("Beginning hyperparameterization!")
    wandb.agent(sweep_id, train)

    print("Sweep complete!")
