#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Vanilla gradient saliency map generation code
"""

import os, gc, glob
import bp_configs
import bp_models
import bp_utility
import numpy as np
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from random import randrange
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import Score, CategoricalScore, BinaryScore
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.models import Model
from matplotlib.colors import LogNorm

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 22})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def grad_cam_analysis(model, input_image, name, pos):
    plt.rcParams.update({'font.size': 28})  
    vars = ['ref', 't', 'q', 'u', 'v']
    titles=['3+_5 Grad-CAM', 'Reflectivity', 'Temperature', 'Specific Humidity', 'U-Wind', 'V-Wind']
    colors=['gist_ncar', 'Blues', 'Purples', 'Oranges', 'Greens'] 

    new_input_image = tf.convert_to_tensor(input_image[np.newaxis, :], dtype=tf.float32) 

    with tf.GradientTape() as tape:
        tape.watch(new_input_image)
        output = model(new_input_image) 

    gradients = tape.gradient(output, new_input_image)
    heatmap = np.squeeze(np.max(np.abs(gradients), axis=-1))

    # Plot the heatmap for the region of interest
    f, ax = plt.subplots(nrows=1, ncols=bp_configs.CHANNELS+1, figsize=(32,6))
    heatmap = heatmap.astype(np.float32)
    heatmap[:16,:] = np.nan
    if np.nanmax(heatmap) > 100:
        im = ax[0].imshow(heatmap, cmap='viridis', alpha=1, norm=LogNorm(vmin=1, vmax=np.nanmax(heatmap)))
    else:
        im = ax[0].imshow(heatmap, cmap='viridis', alpha=1)

    ax[0].invert_yaxis()
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(titles[0])
    for i in range(bp_configs.CHANNELS):
        ax[i+1].set_title(titles[i+1])
        ax[i+1].set_xticks([])
        ax[i+1].set_yticks([])
        ax[i+1].axhline(16, linestyle='--', color='black', linewidth=2)
        im = -1
        if i == 0:
            vals = bp_utility.inv_standardize(input_image[:, :, i], vars[i], 'kazr')
            vals[:16,:] = np.nan
            np.save(bp_configs.prod_dir + '/figures/saliency/refl.npy', vals)
            im = ax[i+1].imshow(vals, \
                        vmin=bp_configs.DATA_RANGE['kazr'][vars[i]][0], vmax=bp_configs.DATA_RANGE['kazr'][vars[i]][1], cmap=colors[i])
        else:
            vals = bp_utility.inv_standardize(input_image[:, :, i], vars[i], 'kazr')
            vals[vals <= bp_configs.DATA_RANGE['kazr'][vars[i]][0]] = np.nan
            im = ax[i+1].imshow(vals, cmap=colors[i])
        plt.colorbar(im, fraction=0.045, ax=ax[i+1])
        ax[i+1].invert_yaxis()
    plt.suptitle('Grad-CAM Features (' + name + ' #' + str(pos) + ')')
    plt.tight_layout()
    fig1 = plt.gcf()
    # plt.show()

    np.save(bp_configs.prod_dir + '/figures/saliency/heatmap.npy', heatmap)
    fig1.savefig(bp_configs.prod_dir + '/figures/saliency/' + name + '_' + str(pos) + '_' + str(bp_configs.CHANNELS) + '_chan')
    return True



def eval_downfill_errors(dsz, path1, use_epoch, model, site_date, id):

    def calc_errors_for_path(path, name):
        print("Calculating errors for", path)
        test_data = np.double(np.load(path))

        print("TRUTH ORIG", np.asarray(test_data).shape)
        with_data = []
        for item in test_data:
            if (np.count_nonzero(item[:dsz,:,0] <= -1) / (item[:dsz,:,0].shape[0] * item[:dsz,:,0].shape[1])) <= 1:
                if bp_configs.CHANNELS == 1:
                    with_data.append(item[:,:,0])
                else:
                    with_data.append(item)
        test_data = copy.deepcopy(with_data)
        if bp_configs.CHANNELS == 1:
            test_data = np.expand_dims(test_data, axis=3)
        print("TRUTH WITH DATA",  np.asarray(test_data).shape)

        buf_size = 8
        sz = bp_configs.SIZE['downfill'][1]
        #get the ground truth:
        truth = []
        for sample in test_data:
            truth.append(sample[:dsz,:,:])
        truth = np.array(truth)

        truth = truth[:,:,:,:bp_configs.CHANNELS] # truth fix
        mask = np.zeros((sz,sz))
        mask[:dsz,:] = 1.0
        
        buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
        mask[dsz:dsz+buf_size,:] = buf[:,np.newaxis]
        mask = mask[:,:,np.newaxis]
        tmp_test_data = []
        for i in range(len(test_data)):
            sample = test_data[i]
            sample[:dsz,:,0] = -1.0
            if bp_configs.CHANNELS > 1:
                if 'era5' in name:
                    sample[:dsz,:,1] = -1.0
                else:
                    sample[:dsz,:,1] = 0.0
                sample[:dsz,:,2] = -1.0
            if bp_configs.CHANNELS == 5:
                sample[:dsz,:,3] = -1.0
                sample[:dsz,:,4] = -1.0
            tmp_test_data.append(np.concatenate((sample,mask,np.random.normal(0,0.5,mask.shape)),axis=2))
        test_data = tmp_test_data
        

        # data = np.array(test_data)[:,:,:,:2]
        # unet3p_refl = bp_models.unet3plus((*bp_configs.SIZE['downfill'], 2), 1, config=bp_configs.config_defaults, depth=bp_configs.config_defaults['depth'], training=True, clm=False)
        # unet3p_refl.load_weights(bp_configs.prod_dir + 'downfill_3net/128_1chan_era5_nsa_oli_10km_dsv_long_hybrid/' + 'epoch_0501' + '/variables/variables').expect_partial()
        # unet3p_refl_pred = unet3p_refl.predict(data, verbose=1, batch_size=1)[0]

        unet3p_dsv = bp_models.unet3plus((*bp_configs.SIZE['downfill'],bp_configs.CHANNELS+1), bp_configs.CHANNELS, config=bp_configs.config_defaults, \
                                            depth=bp_configs.config_defaults['depth'], training=True, clm=False)
        # unet3p_dsv = enable_dropout(unet3p_dsv)            
        unet3p_dsv.load_weights(bp_configs.prod_dir + 'downfill_3net/128_5chan_era5_nsa_oli_10km_dsv_long_hybrid/' + 'epoch_0501' + '/variables/variables').expect_partial()
        unet3p_dsv_pred = unet3p_dsv.predict(np.array(test_data)[:,:,:,:bp_configs.CHANNELS+1], verbose=1, batch_size=1)[0]

        cloud_mask = bp_configs.CLOUD_MASK

        # unetpp_pred = unetpp_pred[:,:dsz,:,:]
        # ref_mask_unetpp = unetpp_pred[:,:,:,0]<cloud_mask
        # unet3p_refl_pred = unet3p_refl_pred[:,:dsz,:,:]
        # ref_mask_unet3p = unet3p_refl_pred[:,:,:,0]<cloud_mask
        unet3p_dsv_pred = unet3p_dsv_pred[:,:dsz,:,:]
        ref_mask_unet3p_dsv = unet3p_dsv_pred[:,:,:,0]<cloud_mask
        
        # unetpp_pred[:,:,:,0][ref_mask_unetpp] = -1.0
        # unet3p_refl_pred[:,:,:,0][ref_mask_unet3p] = -1.0
        unet3p_dsv_pred[:,:,:,0][ref_mask_unet3p_dsv] = -1.0
        #else:
        for i in range(bp_configs.CHANNELS):
            if i == 0 or i == 2 or 'era5' in name:
                # unetpp_pred[:,:,:,i][ref_mask_unetpp] = -1.0
                # unet3p_refl_pred[:,:,:,i][ref_mask_unet3p] = -1.0
                unet3p_dsv_pred[:,:,:,i][ref_mask_unet3p_dsv] = -1.0
            else:
                # unetpp_pred[:,:,:,i][ref_mask_unetpp] = 0
                # unet3p_refl_pred[:,:,:,i][ref_mask_unet3p] = 0
                unet3p_dsv_pred[:,:,:,i][ref_mask_unet3p_dsv] = 0

        stop_flag = grad_cam_analysis(unet3p_dsv, np.array(test_data)[int(id),:,:,:bp_configs.CHANNELS+1], site_date, int(id))

        # Feature Visualization
        # count = 0
        # while(True):
        #     rand = randrange(0, len(test_data))
        #     print(id)
        #     stop_flag = grad_cam_analysis(unet3p_refl, np.array(test_data)[rand,:,:,:bp_configs.CHANNELS+1], site_date, rand)
        #     count += 1
        #     if count > 4:
        #         break

            # if stop_flag:
            #     break

        tf.keras.backend.clear_session()
        # del unetpp;gc.collect()
        # del unet3p_refl;gc.collect()
        del unet3p_dsv;gc.collect()

    calc_errors_for_path(path1, bp_configs.RUN_CASE)

def compute_error_metrics(filepath, name, epoch, model, id):
    downfill_sizes = bp_configs.DOWNFILL_SIZES

    errors = []
    for ds in downfill_sizes:
        errors.append(eval_downfill_errors(ds, filepath, epoch, model, name, id))
        gc.collect()

if __name__ == '__main__':
    data_paths = bp_utility.path_builder()
    # chan1_ids = [156, 216, 431, 172, 130, 105, 167, 261,147,136,21,283,95,64,249,236,400,40,253,394,238,305,251,14,165,466,59,464,79,199,317,12]
    # count = 0
    # for i, path in enumerate(data_paths):
    #     if os.path.isfile(bp_configs.data_dir + '/test_set/test_set_' + path + '_kazr.npy'):
    #         print("\n\nTesting on", path)
    #         compute_error_metrics(bp_configs.data_dir + '/test_set/test_set_' + path + '_kazr.npy', path, 'epoch_0501', 'l1', count)
    #         count+=1
    #         gc.collect()
    #         # break

    count = 0
    files = glob.glob("Z:/data/transfer/short/saliency/*.png")
    for i, path in enumerate(files):
        print(path[-19:-8], path[-7:-4])
        # print("\n\nTesting on", path)
        compute_error_metrics(bp_configs.data_dir + '/test_set/test_set_' + path[-19:-8] + '_kazr.npy', path[-19:-8], 'epoch_0501', 'l1',  path[-7:-4])
        count+=1
        gc.collect()
        # break

    # count = 0
    # path = 'nsa_2013_11'
    # compute_error_metrics(bp_configs.data_dir + '/test_set/test_set_' + path + '_kazr.npy', path, 'epoch_0501', 'l1', 431)
    # gc.collect()
    #         # break


    print("All Tests Complete!")