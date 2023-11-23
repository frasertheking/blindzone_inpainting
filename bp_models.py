#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: UNet model definitions and loss functions
"""

import bp_configs
import tensorflow as tf
import tensorflow.keras as k
import keras.backend as Kb
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Lambda, concatenate
from keras.models import Model
from scipy.signal import convolve2d as conv
import tensorflow.keras.backend as K

#### Custom Loss Functions
def blind_MAE(y_true, y_pred):
    """
    Custom weighted MAE loss function based on the input filters and
    using only the refl channel.
    """
    filt = y_true[:,:,:,bp_configs.CHANNELS]
    y_true = y_true[:,:,:,:1]
    y_pred = y_pred[:,:,:,:1]
    mae = tf.math.abs(y_true-y_pred)
    if bp_configs.CHANNELS == 3:
        filt = tf.stack([filt,filt,filt],axis=3)
    elif bp_configs.CHANNELS == 5:
        filt = tf.stack([filt,filt,filt,filt,filt],axis=3)
    else:
        filt = tf.stack([filt],axis=3)
    weighted_mae = tf.reduce_mean(mae*filt)/tf.reduce_mean(filt)
    return weighted_mae

def mae(y_true, y_pred):
    """
    Calculate mean absolute error between two prediction sets.
    """
    mae = tf.math.abs(y_true - y_pred)
    return mae

def iou(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    
    y_pred = tf.where(y_pred > -0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    y_true = tf.where(y_true > -0.5, tf.ones_like(y_true), tf.zeros_like(y_true))

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return K.mean(1 - ssim_value, axis=0)


def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """

    filt = y_true[:,:,:,bp_configs.CHANNELS]
    y_true = y_true[:,:,:,:bp_configs.CHANNELS]
    filt = tf.stack([filt,filt,filt], axis=3)
    y_true = y_true[:,:,:,:1]
    y_pred = y_pred[:,:,:,:1]

    y_pred = tf.where(y_pred > -0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    y_true = tf.where(y_true > -0.5, tf.ones_like(y_true), tf.zeros_like(y_true))

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice_score = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    weighted_dice = tf.reduce_mean(dice_score*filt)/tf.reduce_mean(filt)

    return weighted_dice

def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """

    filt = y_true[:,:,:,bp_configs.CHANNELS]
    y_true = y_true[:,:,:,:1]
    y_pred = y_pred[:,:,:,:1]
    filt = tf.stack([filt,filt,filt], axis=3)

    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jacard_loss = iou_loss(y_true, y_pred)
    mae_loss = mae(y_true, y_pred)

    weighted_mae_loss =  tf.reduce_mean(mae_loss*filt)/tf.reduce_mean(filt)
    weighted_ms_ssim_loss =  tf.reduce_mean(ms_ssim_loss*filt)/tf.reduce_mean(filt)
    weighted_jacard_loss =  tf.reduce_mean(jacard_loss*filt)/tf.reduce_mean(filt)

    return weighted_mae_loss + weighted_ms_ssim_loss + weighted_jacard_loss

# Additional preprocessing helpers (customiable based on number of channels)
def merge_output(xin, x, num_channels):
    filt = xin[:,:,:,num_channels]
    nchan = num_channels
    outputs = []
    for i in range(nchan):
        merged = filt*x[:,:,:,i] + (1.0-filt)*xin[:,:,:,i]
        outputs.append(Kb.expand_dims(merged))
    return Lambda(lambda x: x)(concatenate(outputs))

def preprocess(x, num_channels):
    filt = tf.floor(x[:,:,:,num_channels])
    x0 = x[:,:,:,0]*(1-filt) - filt
    if num_channels == 2:
        x1 = x[:,:,:,1]*(1-filt) - filt
        x = tf.stack([x0,x1,x[:,:,:,2]],axis=3)
        return Lambda(lambda x: x)(x)
    elif num_channels == 3:
        x1 = x[:,:,:,1]*(1-filt) - filt
        x2 = x[:,:,:,2]*(1-filt) - filt
        x = tf.stack([x0,x1,x2,x[:,:,:,3]],axis=3)
        return Lambda(lambda x: x)(x)
    elif num_channels == 4:
        x1 = x[:,:,:,1]*(1-filt) - filt
        x2 = x[:,:,:,2]*(1-filt) - filt
        x3 = x[:,:,:,3]*(1-filt) - filt
        x = tf.stack([x0,x1,x2,x3,x[:,:,:,4]],axis=3)
        return Lambda(lambda x: x)(x)
    if num_channels == 5:
        if bp_configs.USE_DOP_SPW:
            x1 = x[:,:,:,1]*(1-filt)
            x2 = x[:,:,:,2]*(1-filt) - filt
        else:
            x1 = x[:,:,:,1]*(1-filt) - filt
            x2 = x[:,:,:,2]*(1-filt) - filt
            x3 = x[:,:,:,3]*(1-filt) - filt
            x4 = x[:,:,:,4]*(1-filt) - filt
        x = tf.stack([x0,x1,x2,x3,x4,x[:,:,:,5]],axis=3)
        return Lambda(lambda x: x)(x)
    else:
        x = tf.stack([x0,x[:,:,:,num_channels]],axis=3)
        return Lambda(lambda x: x)(x)       

# Custom 2D convolutional block
def conv(x, channels, filter_size=3):
    x = Conv2D(channels, (filter_size,filter_size), padding='same', activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    return x

def unetpp(INPUT_SIZE, base_channels=8, levels=7, growth=2):
    """
    UNet++ model as defined in Geiss and Hardin, 2021.
    Includes nested skip connections. This is used for comparison to the
    3Net+ model defined later below.
    """
    xin = Input(INPUT_SIZE)
    xpre = preprocess(xin, INPUT_SIZE[2]-1)
    net = []
    for lev in range(levels):
        if lev == 0:
            net.append([concatenate([xpre,conv(xpre,base_channels)])])
        else:
            net_layer = []
            for proc in range(lev+1):
                inputs = []
                if proc < lev:
                    inputs.append(MaxPooling2D((2,2))(net[lev-1][proc]))
                if proc > 0:
                    inputs.append(UpSampling2D((2,2))(net_layer[proc-1]))
                    inputs.append(net[lev-1][proc-1])
                if len(inputs) > 1:
                    inputs = concatenate(inputs)
                else:
                    inputs = inputs[0]
                output = conv(inputs,int(base_channels*growth**(lev-proc)))
                if proc>0:
                    output = concatenate([output,net[lev-1][proc-1]])
                net_layer.append(output)
            net.append(net_layer)
    x = conv(net[-1][-1], base_channels*levels)
    xout = Conv2D(bp_configs.CHANNELS,(1,1), activation='tanh', padding='same')(x)
    xout = merge_output(xin, xout, INPUT_SIZE[2]-1)
    cnn = Model(xin, xout)
    return cnn

"""
Custom UNet3+ with Deep Supervision & Dropout along with associated helpers
"""
def dot_product(seg, cls):
    b, h, w, n = k.backend.int_shape(seg)
    seg = tf.reshape(seg, [-1, h * w, n])
    final = tf.einsum("ijk,ik->ijk", seg, cls)
    final = tf.reshape(final, [-1, h, w, n])
    return final

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

class CLSMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CLSMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CLSMask, self).build(input_shape)

    def call(self, inputs, **kwargs):
        cls, x = inputs
        cls_reshaped = tf.reshape(cls, (-1, 1, 1, 1))
        cls_expanded = tf.broadcast_to(cls_reshaped, tf.shape(x))
        minus_ones = tf.ones_like(x) * -1
        product = cls_expanded * x
        result = product + (1 - cls_expanded) * minus_ones
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def unet3plus(input_shape, output_channels, config, depth=4, training=False, clm=False):

    """ Prep """
    interp = config['interpolation']
    input_layer = k.layers.Input(shape=input_shape, name="input_layer")
    xpre = preprocess(input_layer, output_channels)

    """ Encoder """
    encoders = []
    for i in range(depth+1):
        if i == 0:
            e = conv_block(xpre, config['filters']*(2**i), kernel_size=(config['kernel_size'], config['kernel_size']), l2_reg=config['l2_reg'])
        else:
            e = k.layers.MaxPool2D(pool_size=(2, 2))(encoders[i-1])
            e = k.layers.Dropout(config['dropout'])(e, training=True)
            e = conv_block(e, config['filters']*(2**i), kernel_size=(config['kernel_size'], config['kernel_size']), l2_reg=config['l2_reg'])
            # if i == depth:
            #     e._name='bottleneck'
        encoders.append(e)

    """ Classifier """
    cls = -1
    if clm:
        cls = k.layers.Dropout(rate=config['dropout'])(encoders[len(encoders)-1])
        cls = k.layers.Conv2D(2, kernel_size=(1, 1), padding="same", strides=(1, 1))(cls)
        cls = k.layers.GlobalMaxPooling2D()(cls)
        cls = k.activations.sigmoid(cls)
        cls = tf.argmax(cls, axis=-1)
        cls = cls[..., tf.newaxis]
        cls = tf.cast(cls, dtype=tf.float32)

    """ Middle """
    cat_channels = config['filters']
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
                e_d = k.layers.Dropout(config['dropout'])(e_d, training=True)
                e_d = conv_block(e_d, cat_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, l2_reg=config['l2_reg'])
            elif d == e+1:
                e_d = conv_block(encoders[e], cat_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, l2_reg=config['l2_reg'])
            elif e+1 == len(encoders):
                e_d = k.layers.UpSampling2D(size=(2**(e+1-d), 2**(e+1-d)), interpolation=interp)(encoders[e])
                e_d = k.layers.Dropout(config['dropout'])(e_d, training=True)
                e_d = conv_block(e_d, cat_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, l2_reg=config['l2_reg'])
            else:
                e_d = k.layers.UpSampling2D(size=(2**(e+1-d), 2**(e+1-d)), interpolation=interp)(decoders[decoder_pos-1])
                e_d = k.layers.Dropout(config['dropout'])(e_d, training=True)
                e_d = conv_block(e_d, cat_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, l2_reg=config['l2_reg'])
                decoder_pos -= 1
            loc_dec.append(e_d)
        de = k.layers.concatenate(loc_dec)
        de = conv_block(de, upsample_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, l2_reg=config['l2_reg'])
        decoders.append(de)

    """ Final """
    d1 = decoders[len(decoders)-1]
    d1 = conv_block(d1, output_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, is_bn=False, is_relu=False, l2_reg=config['l2_reg'])
    outputs = [d1]

    """ Deep Supervision """
    if training:
        for i in reversed(range(len(decoders))):
            if i == 0:
                e = conv_block(encoders[len(encoders)-1], output_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, is_bn=False, is_relu=False, l2_reg=config['l2_reg'])
                e = k.layers.UpSampling2D(size=(2**(len(decoders)-i), 2**(len(decoders)-i)), interpolation=interp)(e)
                outputs.append(e)
            else:
                d = conv_block(decoders[i - 1], output_channels, kernel_size=(config['kernel_size'], config['kernel_size']), n=1, is_bn=False, is_relu=False, l2_reg=config['l2_reg'])
                d = k.layers.UpSampling2D(size=(2**(len(decoders)-i), 2**(len(decoders)-i)), interpolation=interp)(d)
                outputs.append(d)
        # outputs.append(cls)

    """ Classifier and Deep Supervision Cont'd """
    if clm:
        print("THIS FEATURE IS DEACTIVATED RIGHT NOW!")
        out1 = CLSMask()([cls, outputs[0][:,:,:,:1]])
        out2 = CLSMask()([cls, outputs[0][:,:,:,1:2]])
        out3 = CLSMask()([cls, outputs[0][:,:,:,2:3]])
        outputs[0] = k.layers.concatenate([out1, out2, out3])
    outputs[0] = merge_output(input_layer, k.activations.linear(outputs[0]), output_channels)

    if training:
        for i in range(len(outputs)):
            if i == 0:
                continue
            d_e = outputs[i]
            if clm:
                out1 = CLSMask()([cls, outputs[i][:,:,:,:1]])
                out2 = CLSMask()([cls, outputs[i][:,:,:,1:2]])
                out3 = CLSMask()([cls, outputs[i][:,:,:,2:3]])
                d_e = k.layers.concatenate([out1, out2, out3])
            outputs[i] = merge_output(input_layer, k.activations.linear(d_e), output_channels)

    return tf.keras.Model(inputs=input_layer, outputs=outputs, name='UNet3Plus')