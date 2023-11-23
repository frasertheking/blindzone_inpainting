#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: precompute MCDropout tests for all models and save output for later analysis
"""

import os, gc
import bp_configs
import bp_schemes
import bp_models
import bp_utility
import numpy as np
import copy
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from multiprocess import Pool
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import cm

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 22})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    return val, val2, n

def grad_cam_analysis(model, input_image, name, i):
    plt.rcParams.update({'font.size': 16})  
    vars = ['ref', 't', 'q', 'u', 'v']
    titles=['Grad-CAM++', 'Reflectivity (dBZ)', 'Temperature (K)', 'Specific Humidity (kg/kg)', 'U-Wind (m/s$^2$)', 'V-Wind (m/s$^2$)']
    colors=['gist_ncar', 'Blues', 'Purples', 'Oranges', 'Greens']

    def model_modifier(current_model):
        target_layer = current_model.get_layer('conv2d_9')
        return tf.keras.Model(inputs=current_model.inputs, outputs=target_layer.output)
 
    gradcam = GradcamPlusPlus(model, model_modifier=model_modifier, clone=True)

    # Compute the GradCAM heatmap for the region of interest
    cam = gradcam([CategoricalScore([0])], input_image[np.newaxis, :], penultimate_layer=-1)
    cam = np.squeeze(cam)

    # Normalize the heatmap
    heatmap = np.uint8(cm.viridis(cam / np.max(cam))[..., :3] * 255)

    # Plot the heatmap for the region of interest
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(24, 4))
    im = ax[0].imshow(heatmap, cmap='viridis', alpha=1)
    ax[0].invert_yaxis()
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(titles[0])
    plt.colorbar(im,  fraction=0.045, ax=ax[0])
    for i in range(5):
        ax[i+1].set_title(titles[i+1])
        ax[i+1].set_xticks([])
        ax[i+1].set_yticks([])
        ax[i+1].axhline(16, linestyle='--', color='black', linewidth=2)
        im = -1
        if i == 0:
            im = ax[i+1].imshow(bp_utility.inv_standardize(input_image[:, :, i], vars[i], 'kazr'), \
                        vmin=bp_configs.DATA_RANGE['kazr'][vars[i]][0], vmax=bp_configs.DATA_RANGE['kazr'][vars[i]][1], cmap=colors[i])
        else:
            im = ax[i+1].imshow(bp_utility.inv_standardize(input_image[:, :, i], vars[i], 'kazr'), cmap=colors[i])
        plt.colorbar(im, fraction=0.045, ax=ax[i+1])
        ax[i+1].invert_yaxis()
    plt.suptitle('Grad-CAM Features (' + name + ' #' + str(i) + ')')
    plt.tight_layout()
    plt.show()

def visualize_features(model, x):
    plt.rcParams.update({'font.size': 75})
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    successive_feature_maps = visualization_model.predict(x[20:21,:,:])# x[63:64,:,:])
    layer_names = [layer.name for layer in model.layers]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
            print(n_features, size)
            
            # We will tile our images in this matrix
            factor, factor_2, _ = factor_int(n_features)
            if factor < 4:
                continue
            display_grid = np.zeros((factor*size, factor_2*size))
            
            # Postprocess the feature to be visually palatable
            count = 0
            for i in range(factor):
                for j in range(factor_2):
                    x  = feature_map[0, :, :, count]
                    # x -= x.mean()
                    # x /= x.std ()
                    # x *=  64
                    # x += 128
                    # x  = np.clip(x, 0, 255).astype('uint8')
                    # Tile each filter into a horizontal grid
                    display_grid[i * size : (i + 1) * size, j * size : (j + 1) * size] = x
                    count+=1
            
            # Display the grid
            scale = 50. / factor
            plt.figure( figsize=(int(scale * factor_2), int(scale * factor)) )
            plt.title ( layer_name )
            plt.grid  ( False )
            plt.imshow( display_grid, aspect='auto', cmap='viridis' )
            plt.gca().invert_yaxis()

            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            for i in range(int(size*factor_2)):
                if i%size == 0:
                    plt.axvline(i, linewidth=7, color='white')

            for i in range(int(size*factor)):
                if i%size == 0:
                    plt.axhline(i, linewidth=7, color='white')

            plt.savefig("features/feature_visualization_" + layer_name + ".png")

def inpaint(data, mask, func):
    data = np.copy(data)
    mask = np.copy(mask)
    for i in range(bp_configs.CHANNELS):
        cdat = data[:,:,i]
        cdat = cdat*0.5+0.5
        cdat = func(cdat,np.copy(mask))
        cdat = cdat*2.0-1.0
        if i == 0:
            ref_mask = cdat<-0.5
            cdat[ref_mask] = -1.0
        if i == 1:
            cdat[ref_mask] = 0.0
        if i == 2:
            cdat[ref_mask] = -1.0
        data[:,:,i] = cdat
    return data
    
def emd(x, y, inst='kazr'):
    rng = [[-10,40],[-12,12],[0,5]]
    EMD = []
    norm = x.shape[1]*x.shape[2]
    nbin = 64
    for i in range(x.shape[0]):
        femd = []
        for j in range(x.shape[-1]):
            hy = np.histogram(y[i,:,:,j], bins=nbin, range=rng[j])[0]
            hx = np.histogram(x[i,:,:,j], bins=nbin, range=rng[j])[0]
            cy = np.cumsum(hy)/norm
            cx = np.cumsum(hx)/norm
            femd.append(100*np.abs(np.sum(cx-cy))/nbin)
        EMD.append(femd)
    return np.mean(np.array(EMD),axis=0)
    
def psd(x):
    x = x[:,:int(2**np.floor(np.log2(x.shape[1]))),...]
    fft = np.abs(np.fft.rfft(x,axis=1))
    mnfft = np.nanmean(fft,axis=2)
    mnpsd = 10*np.log10(mnfft**2)
    mnpsd[np.abs(mnpsd)>10000] = np.nan
    mnpsd = np.nanmean(mnpsd,axis=0)
    return mnpsd[1:-1]

def eval_downfill_errors(dsz, path1, use_epoch, model, site_date):
    # Enable Dropout for Monte Carlo
    def enable_dropout(model):
        model_config = model.get_config()
        pos = 0
        for layer in model_config['layers']:
            if 'dropout' in layer['name']:
                model_config['layers'][pos]['inbound_nodes'][0][0][-1]['training'] = True
            pos += 1
        return tf.keras.models.Model.from_config(model_config)

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
        truth = []
        for sample in test_data:
            truth.append(sample[:dsz,:,:])
        truth = np.array(truth)

        truth = truth[:,:,:,:bp_configs.CHANNELS] # truth fix

        #make a binary mask:
        mask = np.zeros((sz,sz))
        mask[:dsz,:] = 1.0
        
        p = Pool(12)
        marching_avgs = p.map(lambda x: inpaint(x,mask,bp_schemes.marching_avg),np.array(test_data)[:,:,:,:bp_configs.CHANNELS])
        marching_avgs = np.array(marching_avgs)[:,:dsz,:,:]
        repeats = p.map(lambda x: inpaint(x,mask,bp_schemes.repeat),np.array(test_data)[:,:,:,:bp_configs.CHANNELS])
        repeats = np.array(repeats)[:,:dsz,:,:]
        p.close()
        
        #prep the inputs for the CNNs:
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
            if bp_configs.CHANNELS == 4:
                sample[:dsz,:,3] = -1.0
            if bp_configs.CHANNELS == 5:
                sample[:dsz,:,3] = -1.0
                sample[:dsz,:,4] = -1.0
            tmp_test_data.append(np.concatenate((sample,mask,np.random.normal(0,0.5,mask.shape)),axis=2))
        test_data = tmp_test_data

        # Used for four channel case
        test_data = np.delete(np.array(test_data), 3, axis=3)
        
        N_TESTS = 50 #bp_configs.N_MC_TESTS
        unetpp_preds = []
        unet3p_refl_preds = []
        unet3p_dsv_preds = []
        for i in range(0, N_TESTS, 1):
            print("\n################")
            print("On iteration", i)
            print("################\n")
            # random.seed()
            #the l1 case
            # unetpp = bp_models.unetpp((*bp_configs.SIZE['downfill'],bp_configs.CHANNELS+1),base_channels=8,levels=7,growth=2)
            # # unetpp = enable_dropout(unetpp)
            # unetpp.load_weights(bp_configs.prod_dir + 'downfill_l1/128_5chan_era5_nsa_oli_10km_unetpp/' + use_epoch + '/variables/variables').expect_partial()
            # unetpp_pred = unetpp.predict(np.array(test_data)[:,:,:,:bp_configs.CHANNELS+1], verbose=1, batch_size=1)

            # data = np.array(test_data)[:,:,:,:2]
            # unet3p_refl = bp_models.unet3plus((*bp_configs.SIZE['downfill'], 2), 1, config=bp_configs.config_defaults, depth=bp_configs.config_defaults['depth'], training=True, clm=False)
            # unet3p_refl = enable_dropout(unet3p_refl)
            # unet3p_refl.load_weights(bp_configs.prod_dir + 'downfill_3net/128_1chan_era5_nsa_oli_10km_dsv_long/' + 'epoch_2001' + '/variables/variables').expect_partial()
            # unet3p_refl_pred = unet3p_refl.predict(data, verbose=1, batch_size=1)[0]

            unet3p_dsv = bp_models.unet3plus((*bp_configs.SIZE['downfill'],bp_configs.CHANNELS+1), bp_configs.CHANNELS, config=bp_configs.config_defaults, \
                                              depth=bp_configs.config_defaults['depth'], training=True, clm=False)
            unet3p_dsv = enable_dropout(unet3p_dsv)            
            unet3p_dsv.load_weights(bp_configs.prod_dir + 'downfill_3net/128_4chan_era5_nsa_oli_10km_dsv/' + 'epoch_0501' + '/variables/variables').expect_partial()
            unet3p_dsv_pred = unet3p_dsv.predict(np.array(test_data)[:,:,:,:bp_configs.CHANNELS+1], verbose=1, batch_size=1)[0]

            cloud_mask = bp_configs.CLOUD_MASK

            # print(unet3p_refl_pred.shape)
            # plt.imshow(unet3p_refl_pred[25,:,:,0])
            # plt.show()
            # plt.imshow(unet3p_refl_pred[50,:,:,0])
            # plt.show()
            # plt.imshow(unet3p_refl_pred[100,:,:,0])
            # plt.show()
            # plt.imshow(unet3p_refl_pred[150,:,:,0])
            # plt.show()

            # print(unetpp_pred.shape)
            # print(unet3p_refl_pred.shape)
            # print(unet3p_dsv_pred.shape)
            # sys.exit()

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

            # for i in range(cnn_pred.shape[0]):
            #     bp_plotting.plot(cnn_pred[i,:,:,:bp_configs.CHANNELS], name,fname=bp_configs.image_path + "/test_figures/sample_asd" + str(dsz) + '_' + str(i) + '.png')
                    
            # unetpp_preds.append(unetpp_pred)
            # unet3p_refl_preds.append(unet3p_refl_pred)
            unet3p_dsv_preds.append(unet3p_dsv_pred)
            
            # Feature Visualization
            #visualize_features(unet3p_dsv, np.array(test_data)[:,:,:,:bp_configs.CHANNELS+1])
            # for i in range(180):
            #     from random import randrange, uniform
            #     rand = randrange(0, 180)
            #     grad_cam_analysis(unet3p_dsv, np.array(test_data)[rand,:,:,:bp_configs.CHANNELS+1], site_date, rand)
                # break

            # del unetpp;gc.collect()
            # del unet3p_refl;gc.collect()
            del unet3p_dsv;gc.collect()
            # sys.exit()

        # Save intermediary
        # np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_unetpp_' + str(bp_configs.N_MC_TESTS) + '.npy', np.asarray(unetpp_preds)[:,:,:,:,0])
        # np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_unet3p_refl_' + str(bp_configs.N_MC_TESTS) + '.npy', np.asarray(unet3p_refl_preds)[:,:,:,:])
        # np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_unet3p_hybrid_real_' + str(bp_configs.N_MC_TESTS) + '.npy', np.asarray(unet3p_dsv_preds)[:,:,:,:,0])
        np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_unet3p_4chan_' + str(bp_configs.N_MC_TESTS) + '.npy', np.asarray(unet3p_dsv_preds)[:,:,:,:,0])
        # np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_marching.npy', marching_avgs[:,:,:,0])
        # np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_repeating.npy', repeats[:,:,:,0])
        # np.save(bp_configs.prod_dir + '/prod_eval/predictions/' + site_date + '_truth.npy', truth[:,:,:,0])

        return None, None, None, None, None, None, None
    
    # rep_pod, march_pod, l1_pod, rep_far, march_far, l1_far, rep_hss, march_hss, l1_hss = calc_errors_for_path(path1, bp_configs.RUN_CASE)
    MAE, MSE, EMD, PSD, PSDt, true_psd, true_psdt = calc_errors_for_path(path1, bp_configs.RUN_CASE)

    # print()
    # print("MODEL", model)
    # print("MAE", MAE)
    # print("MSE", MSE)
    # print("EMD", EMD)
    # print()
    # print("PSD", PSD)
    # print("True PSD", true_psd)
    # print()
    # print("PSDt", PSDt)
    # print("True PSD", true_psdt)
    return MAE, MSE, EMD, PSD, PSDt, true_psd, true_psdt

def compute_error_metrics(filepath, name, epoch, model, sub_model):
    downfill_sizes = bp_configs.DOWNFILL_SIZES
    errors = []
    for ds in downfill_sizes:
        errors.append(eval_downfill_errors(ds, filepath, epoch, model, name))
        gc.collect()
    # np.save(bp_configs.prod_dir + '/prod_eval/eval_' + name + '.npy', errors)

if __name__ == '__main__':
    data_paths = bp_utility.path_builder()
    for i, path in enumerate(data_paths):
        if os.path.isfile(bp_configs.data_dir + 'test_set/test_set_' + path + '_kazr.npy'):
            print("\n\nTesting on", path)
            compute_error_metrics(bp_configs.data_dir + 'test_set/test_set_' + path + '_kazr.npy', path, 'epoch_0501', 'l1', '')
            gc.collect()
            # break

    print("All Tests Complete!")