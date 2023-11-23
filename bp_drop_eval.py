#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: This module performs the MCDropout, predicting the blind zone values N times and saving the results
"""

import os
import math
import copy
import bp_utility
import bp_configs
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

models = ['unet3p_hybrid_50', 'unet3p_refl_50', 'unet3p_4chan_50', 'unetpp_50',  'marching', 'repeating'] 

# Dice similarity function
def calc_DICE(y_pred, y_true, k = 1):
    y_pred[y_pred > -60] = k
    y_pred[y_pred < 1] = 0
    y_true[y_true > -60] = k
    y_true[y_true < 1] = 0
    intersection = np.sum(y_pred[y_true==k]) * 2.0
    dice = intersection / (np.sum(y_pred) + np.sum(y_true))
    return dice

def calc_PSD(x):
    x = x[:,:int(2**np.floor(np.log2(x.shape[1]))),...]
    fft = np.abs(np.fft.rfft(x,axis=0))
    mnfft = np.nanmean(fft,axis=1)
    mnpsd = 10*np.log10(mnfft**2)
    mnpsd[np.abs(mnpsd)>10000] = np.nan
    return mnpsd

def calc_EMD(y_pred, y_true):
    rng = [-60,30]
    EMD = []
    norm = y_pred.shape[0]*y_pred.shape[1]
    nbin = 64
    for i in range(y_pred.shape[0]):
        femd = []
        hy = np.histogram(y_true, bins=nbin, range=rng)[0]
        hx = np.histogram(y_pred, bins=nbin, range=rng)[0]
        cy = np.cumsum(hy)/norm
        cx = np.cumsum(hx)/norm
        femd.append(100*np.abs(np.sum(cx-cy))/nbin)
        EMD.append(femd)
    return np.nanmean(np.array(EMD),axis=0)

def is_shallow_snowfall(blindzone_arr, bins_above_bz):
    if (np.nanmax(bins_above_bz) < bp_configs.PRECIPITATION_DBZ) and \
        (np.nanmax(blindzone_arr[:1,:]) >= bp_configs.PRECIPITATION_DBZ):
        return True
    return False

def is_virga(blindzone_arr, bins_above_bz):
    if (np.nanmax(bins_above_bz) >= bp_configs.PRECIPITATION_DBZ) and \
        (np.nanmax(blindzone_arr[:1,:]) < bp_configs.PRECIPITATION_DBZ):
        return True
    return False

def cloud_exists(arr, _loc):
    perc_full = np.count_nonzero(arr > -60) / arr.size
    if perc_full >= bp_configs.PERC_CLOUD:
        return True
    return False

def find_lowest_precip_layer(image_data):
    for row_idx, row in enumerate(image_data):
        if np.any(row > bp_configs.PRECIPITATION_DBZ):
            return row_idx
    return np.nan

def find_cloud_base(image_data):
    for row_idx, row in enumerate(image_data):
        if np.any(row > -60):
            return row_idx
    return np.nan

def find_cloud_top(image_data):
    for row_idx, row in enumerate(image_data[::-1]):
        if np.any(row > -60):
            return len(image_data) - 1 - row_idx
    return np.nan

def compute_cloud_depth(image_data):
    cloud_pixels = np.sum(image_data > -60)
    total_pixels = image_data.size
    depth_ratio = cloud_pixels / total_pixels
    return depth_ratio

def count_cloud_layers(image_data):
    cloud_layers = 0
    in_cloud = False
    for row in image_data:
        if np.any(row > -60):
            if not in_cloud:
                in_cloud = True
                cloud_layers += 1
        else:
            in_cloud = False
    return cloud_layers

def get_cloud_features(image_data):
    cloud_top = find_cloud_top(image_data)
    cloud_base = find_cloud_base(image_data)
    cloud_depth = compute_cloud_depth(image_data)
    cloud_layers = count_cloud_layers(image_data)
    cloud_precip_layer = find_lowest_precip_layer(image_data)

    return cloud_top, cloud_base, cloud_depth, cloud_layers, cloud_precip_layer

def calc_hits_n_misses(func, y_pred, y_test, bins_above_bz):
    hit = 0
    miss = 0
    fa = 0
    cn = 0
    if func(y_test, bins_above_bz) and func(y_pred, bins_above_bz): # hit
        hit += 1
    elif func(y_test, bins_above_bz) and not(func(y_pred, bins_above_bz)): # miss
        miss += 1
    elif not(func(y_test, bins_above_bz)) and func(y_pred, bins_above_bz): # false alarm
        fa += 1
    elif not(func(y_test, bins_above_bz)) and not(func(y_pred, bins_above_bz)): # corr neg
        cn += 1
    return hit, miss, fa, cn

def calc_HSS(hits_and_misses):
    hit = hits_and_misses[0]
    miss = hits_and_misses[1]
    fa = hits_and_misses[2]
    cn = hits_and_misses[3]
    hss = np.nan
    if ( (hit + miss)*(miss + cn) + (hit + fa)*(fa + cn) ) == 0:
        hss = np.nan
    else:
        hss = 2 * ( (hit * cn) - (fa * miss) ) / ( (hit + miss)*(miss + cn) + (hit + fa)*(fa + cn) )
    return hss
  
def calc_MAE(y_pred, y_true):
    return np.nanmean(np.abs(y_pred - y_true))

def calc_RMSE(y_pred, y_true):
    return math.sqrt(np.nanmean((y_pred - y_true)**2))

def stitch_inputs(orig, infilled, perform_inv):
    loc_orig = np.copy(orig[:,:,0])
    loc_orig[:bp_configs.DOWNFILL_SIZES[0],:] = infilled[:,:]
    if perform_inv:
        return bp_utility.inv_standardize(loc_orig, 'ref', 'kazr')
    return loc_orig

def eval_individual_runs(name):
    ds_outer = np.load(bp_configs.data_dir + 'test_set/test_set_' + name + '_kazr.npy', allow_pickle=True)
    ds_truth = np.load(bp_configs.prod_dir + 'prod_eval/predictions/' + name + '_truth.npy', allow_pickle=True)
    print("Number of samples", ds_outer.shape[0])

    MAEs = [[] for i in range(len(models))]
    RMSEs = [[] for i in range(len(models))]
    PSDs = [[] for i in range(len(models))]
    true_PSDs = []
    HSSs = [-1 for i in range(len(models))]
    hits_n_misses = [[] for i in range(len(models))]
    shallow_hits = [[] for i in range(len(models))]
    virga_hits = [[] for i in range(len(models))]
    dices = [[] for i in range(len(models))]
    EMDs = [[] for i in range(len(models))]
    cloud_bases = [[] for i in range(len(models))]
    cloud_lowest_precip_layers = [[] for i in range(len(models))]
    cloud_features = []
    skip_count = 0
    
    print(ds_outer.shape, ds_truth.shape)
    for i in range(ds_outer.shape[0]):
        truth_val = stitch_inputs(ds_outer[i], ds_truth[i], True)
        cloud_feature = get_cloud_features(truth_val)

        # skip = False
        if np.max(truth_val) > bp_configs.MAX_DBZ:
            continue

        filled_vals = [[] for m in range(len(models))]
        filled_stds = []
        bz_filleds = []
        is_there_cloud = []
        for j, model in enumerate(models):
            ds_mod = np.load(bp_configs.prod_dir + 'prod_eval/predictions/' + name + '_' + model + '.npy', allow_pickle=True)
            if j == 3:
                ds_mod = np.squeeze(ds_mod)

            filled = -1
            bz_filled = -1
            if j > 2:
                filled = stitch_inputs(ds_outer[i], ds_mod[i], True)
                bz_filled = bp_utility.inv_standardize(ds_mod[i], 'ref', 'kazr')
            else:
                invs = []
                for k in range(bp_configs.N_MC_TESTS):
                    invs.append(bp_utility.inv_standardize(ds_mod[k][i], 'ref', 'kazr'))
                    # if j == 1:
                    #     print(name, model)
                    #     print(np.nanmean(ds_mod), np.max(ds_mod))
                    #     plt.imshow(bp_utility.inv_standardize(ds_mod[k][i], 'ref', 'kazr'))
                    #     plt.show()

                se = np.squeeze(np.nanstd(np.asarray(invs), axis=0) / math.sqrt(bp_configs.N_MC_TESTS))
                filled_stds.append(stitch_inputs(np.zeros((128,128,1)), se, False))
                se_mask = np.where(se > bp_configs.STD_CUTOFF, np.nan, 1)

                predicted_blind_zone_avg = np.squeeze(np.nanmean(np.asarray(invs), axis=0))
                # print(np.count_nonzero(se_mask < 1), (np.count_nonzero(predicted_blind_zone_avg > -60)))
                if j == 0 and (np.count_nonzero(~np.isnan(se_mask)) < (np.count_nonzero(predicted_blind_zone_avg > -60) / 2)):
                    # print(np.count_nonzero(se_mask > 0), np.count_nonzero(~np.isnan(se_mask)), (np.min(predicted_blind_zone_avg)))
                    skip_count += 1
                    # break
                predicted_blind_zone_avg = np.multiply(predicted_blind_zone_avg, se_mask)
                predicted_blind_zone_avg[predicted_blind_zone_avg < -60] = -60
                predicted_blind_zone_avg[predicted_blind_zone_avg > 30] = 30
                filled = stitch_inputs(bp_utility.inv_standardize(ds_outer[i], 'ref', 'kazr'), predicted_blind_zone_avg, False)
                bz_filled = predicted_blind_zone_avg

            bz_filleds.append(bz_filled)
            is_there_cloud.append(cloud_exists(bz_filled, None))
            filled_vals[j] = (filled)
            cloud_bases[j].append(find_cloud_base(filled))
            cloud_lowest_precip_layers[j].append(find_lowest_precip_layer(filled))

            bins_above_bz = filled[bp_configs.DOWNFILL_CUT_RANGE[0]:bp_configs.DOWNFILL_CUT_RANGE[0]+1,:]
            
            MAEs[j].append(calc_MAE(bz_filled, bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr')))
            RMSEs[j].append(calc_RMSE(bz_filled, bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr')))
            EMDs[j].append(calc_EMD(bz_filled, bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr')))
            dices[j].append(calc_DICE(copy.deepcopy(bz_filled), copy.deepcopy(bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr'))))
            PSDs[j].append(calc_PSD(bz_filled))
            hits_n_misses[j].append(calc_hits_n_misses(cloud_exists, bz_filled, bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr'), None))
            shallow_hits[j].append(calc_hits_n_misses(is_shallow_snowfall, bz_filled, bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr'), bins_above_bz))
            virga_hits[j].append(calc_hits_n_misses(is_virga, bz_filled, bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr'), bins_above_bz))

            if j == 0: # only need to save the true PSD once
                true_PSDs.append(calc_PSD(bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr')))
                cloud_features.append(cloud_feature)

    print("skip count", skip_count)
    hits_n_misses = np.nansum(hits_n_misses, axis=1)
    shallow_hits = np.nansum(shallow_hits, axis=1)
    virga_hits = np.nansum(virga_hits, axis=1)
    for i, model in enumerate(hits_n_misses):
        HSSs[i] = calc_HSS(model)

    print("\nScores:")
    print("MAE:", np.nanmean(MAEs, axis=1))
    print("RMSE:", np.nanmean(RMSEs, axis=1))
    print("EMDs:", np.nanmean(EMDs, axis=1))
    print("Dices:", np.nanmean(dices, axis=1))

    print(hits_n_misses)
    print(shallow_hits)
    print(virga_hits)
    print("\nCloud Bases:", np.nanmean(cloud_bases, axis=1))
    print("\nCloud Lowest Precipitating Layers:", np.nanmean(cloud_lowest_precip_layers, axis=1))
    print(np.nanmean(cloud_features, axis=0))

    return np.nanmean(MAEs, axis=1), np.nanmean(EMDs, axis=1), np.nanmean(RMSEs, axis=1), np.nanmean(dices, axis=1), HSSs, hits_n_misses, true_PSDs, PSDs, shallow_hits, virga_hits, np.nanmean(cloud_features, axis=0), cloud_bases, cloud_lowest_precip_layers

if __name__ == '__main__':
    data_paths = bp_utility.path_builder()
    for i, path in enumerate(data_paths):
        if os.path.isfile(bp_configs.data_dir + 'test_set/test_set_' + path + '_kazr.npy'):
            print("\n\nTesting on", path)
            MAEs, EMDs, RMSEs, dices, HSSs, hits_n_misses, true_PSDs, PSDs, shallow_hits, virga_hits, cloud_features, cloud_bases, cloud_lowest_precip_layers = eval_individual_runs(path)
            np.save(bp_configs.prod_dir + '/prod_eval/eval_' + path + '.npy', [MAEs, EMDs, RMSEs, dices, HSSs, hits_n_misses.flatten(), true_PSDs, PSDs, shallow_hits.flatten(), virga_hits.flatten(), cloud_features, cloud_bases, cloud_lowest_precip_layers])
            # break
