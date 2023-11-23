#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Plotting of various skill metrics of inpainted region accuracy
"""

import os
import math
import glob
import bp_utility
import bp_configs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 22})
from matplotlib.patches import Patch

sites = ['NSA', 'SGP', 'OLI', 'AWR']
colors =['red', 'blue', 'orange', 'black', 'black', 'red', 'blue', 'orange', 'black', 'black', 'red', 'blue', 'orange', 'black', 'black', 'red', 'blue', 'orange']
months = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
# var_type = 1 # 0=MAE 2=EMD

def build_monthly_intercomparisons(loc, units, var_type):
    print("Building Monthly Comparisons for", loc)
    month_vals = [[[] for i in range(7)] for j in range(6)]
    monthly_all_vals = [[] for j in range(6)]

    for file in glob.glob(bp_configs.prod_dir + "/prod_eval/*.npy"):
        basename = os.path.basename(file)

        if not(basename[5:16] in bp_configs.FINAL_PERIODS):
            continue

        site = basename[5:8]
        year = basename[9:13]
        month = int(basename[14:16])
        print("Working on", site, year, month)
        data = np.load(file, allow_pickle=True)

        # if 'oli' in site:
        #     continue

        for i in range(6):
            month_pos = -1
            if (month) == 10:
                month_pos = 0
            elif (month) == 11:
                month_pos = 1
            elif (month) == 12:
                month_pos = 2
            elif (month) == 1:
                month_pos = 3
            elif (month) == 2:
                month_pos = 4
            elif (month) == 3:
                month_pos = 5
            elif (month) == 4:
                month_pos = 6

            month_vals[i][month_pos].append(data[var_type][i])
            monthly_all_vals[i].append(data[var_type][i])

    names = ['3+_5', '3+_1', '3+_4', '++', 'MAR', 'REP']
    colors = ['red', 'blue', 'purple', 'orange', 'green', 'gold']
    fig, ax = plt.subplots(figsize=(14,7))
    plt.title("Monthly Means - " + loc)
    plt.ylabel(loc + " " + units)
    plt.xlabel("Month")

    for i in range(6):
        print(i)
        if i == 2:
            continue
        plt.plot(months, np.vectorize(np.nanmean)(month_vals[i]), linestyle='-', linewidth=2, color=colors[i], label=names[i])
        plt.scatter(months, np.vectorize(np.nanmean)(month_vals[i]), s=30, color=colors[i])
        # plt.fill_between(months, (np.vectorize(np.nanmean)(month_vals[i]) - 1*np.vectorize(np.nanstd)(month_vals[i])/math.sqrt(len(month_vals[i]))), (np.vectorize(np.nanmean)(month_vals[i]) + 1*np.vectorize(np.nanstd)(month_vals[i])/math.sqrt(len(month_vals[i]))), color=colors[i], alpha=0.15)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + "/figures/" + loc + "_monthly.png", transparent=False)

    def moving_average(a, n=2) :
        ret = np.nancumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    all_months = np.arange(len(monthly_all_vals[0])-1)
    fig, ax = plt.subplots(figsize=(14,7))
    plt.title("All Months - " + loc)
    plt.ylabel(loc + " " + units)
    plt.xlabel("Month")
    for i in range(6):
        if i == 2:
            continue
        plt.plot(all_months, moving_average(monthly_all_vals[i]), linestyle='-', linewidth=2,  color=colors[i], label=names[i])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)

    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + "/figures/" + loc + "_all_months.png", transparent=False)

def plot_psds():
    plt.rcParams.update({'font.size': 35})

    vrepeat = []
    vmarch = []
    vunetpp = []
    vunet3p = []
    vunet3p_dsv = []
    vtruth = []

    vrepeat_std = []
    vmarch_std = []
    vunetpp_std = []
    vunet3p_std = []
    vunet3p_dsv_std = []
    vtruth_std = []

    hrepeat = []
    hmarch = []
    hunetpp = []
    hunet3p = []
    hunet3p_dsv = []
    htruth = []

    for file in glob.glob(bp_configs.prod_dir + "/prod_eval/*.npy"):
        print(file)
        
        basename = os.path.basename(file)
        month = int(basename[14:16])

        # if 'oli' in basename:
        #     continue

        if basename[5:16] in bp_configs.FINAL_PERIODS:
            evals = np.load(file, allow_pickle=True)
            vrepeat.append(np.nanmean(np.asarray(evals[7][5][1:]), axis=0))
            vmarch.append(np.nanmean(np.asarray(evals[7][4][1:]), axis=0))
            vunetpp.append(np.nanmean(np.asarray(evals[7][3][1:]), axis=0))
            vunet3p.append(np.nanmean(np.asarray(evals[7][1][1:]), axis=0))
            vunet3p_dsv.append(np.nanmean(np.asarray(evals[7][0][1:]), axis=0))
            vtruth.append(np.nanmean(np.asarray(evals[6][1:]), axis=0))

            vmarch_std.append(np.nanstd(np.asarray(evals[7][4][1:])) / math.sqrt(len(evals[7][4][1:])))
            vunetpp_std.append(np.nanstd(np.asarray(evals[7][3][1:])) / math.sqrt(len(evals[7][3][1:])))
            vunet3p_std.append(np.nanstd(np.asarray(evals[7][1][1:])) / math.sqrt(len(evals[7][1][1:])))
            vunet3p_dsv_std.append(np.nanstd(np.asarray(evals[7][0][1:])) / math.sqrt(len(evals[7][0][1:])))
            vtruth_std.append(np.nanstd(np.asarray(evals[6][1:])) / math.sqrt(len(evals[6][1:])))

    print("MARCH", np.nanmean(np.nanmean(vrepeat, axis=0) - np.nanmean(vtruth, axis=0)))
    print("++", np.nanmean(np.nanmean(vmarch, axis=0) - np.nanmean(vtruth, axis=0)))
    print("3+_4", np.nanmean(np.nanmean(vunetpp, axis=0) - np.nanmean(vtruth, axis=0)))
    print("3+_1", np.nanmean(np.nanmean(vunet3p, axis=0) - np.nanmean(vtruth, axis=0)))
    print("3+_5", np.nanmean(np.nanmean(vunet3p_dsv, axis=0) - np.nanmean(vtruth, axis=0)))

    lw = 5
    x = np.arange(9)
    fig, ax = plt.subplots(figsize=(15,15))
    plt.title("Vertical Power Spectral Density Curves")
    plt.ylabel("Power Spectral Density (dBZ)")
    plt.xlabel("Frequency (km$^{-1}$)")
    # plt.plot(x, np.nanmean(vrepeat, axis=0), linewidth=lw, label="Repeat")
    plt.plot(x, np.nanmean(vmarch, axis=0), linewidth=lw, color='green', label="MAR")
    plt.scatter(x, np.nanmean(vmarch, axis=0), s=100, color='green')
    plt.fill_between(x, np.nanmean(vmarch, axis=0) - np.nanmean(vmarch_std, axis=0), np.nanmean(vmarch, axis=0) + np.nanmean(vmarch_std, axis=0), alpha=0.1, color='green')
   
    plt.plot(x, np.nanmean(vunetpp, axis=0), linewidth=lw, color='orange', label="++")
    plt.scatter(x, np.nanmean(vunetpp, axis=0), s=100, color='orange')
    plt.fill_between(x, np.nanmean(vunetpp, axis=0) - np.nanmean(vunetpp_std, axis=0), np.nanmean(vunetpp, axis=0) + np.nanmean(vunetpp_std, axis=0), alpha=0.1, color='orange')
    
    plt.plot(x, np.nanmean(vunet3p, axis=0), linewidth=lw, color='blue', label="3+_1")
    plt.scatter(x, np.nanmean(vunet3p, axis=0), s=100, color='blue')
    plt.fill_between(x, np.nanmean(vunet3p, axis=0) - np.nanmean(vunet3p_std, axis=0), np.nanmean(vunet3p, axis=0) + np.nanmean(vunet3p_std, axis=0), alpha=0.1, color='blue')
    
    plt.plot(x, np.nanmean(vunet3p_dsv, axis=0), linewidth=lw, color='red', label="3+_5")
    plt.scatter(x, np.nanmean(vunet3p_dsv, axis=0), s=100, color='red')
    plt.fill_between(x, np.nanmean(vunet3p_dsv, axis=0) - np.nanmean(vunet3p_dsv_std, axis=0), np.nanmean(vunet3p_dsv, axis=0) + np.nanmean(vunet3p_dsv_std, axis=0), alpha=0.1, color='red')
    
    plt.plot(x, np.nanmean(vtruth, axis=0), linewidth=lw, color='black', label="KaZR")
    plt.scatter(x, np.nanmean(vtruth, axis=0), s=100, color='black')
    plt.fill_between(x, np.nanmean(vtruth, axis=0) - np.nanmean(vtruth_std, axis=0), np.nanmean(vtruth, axis=0) + np.nanmean(vtruth_std, axis=0), alpha=0.1, color='black')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + "/figures/vert_psd.png")

    # x = np.arange(len(evals[0][4][2].T[0]))
    # fig, ax = plt.subplots(figsize=(12,12))
    # plt.title("Horizontal Reflectivity PSD")
    # plt.ylabel("Power Spectral Density (dBZ)")
    # plt.xlabel("Frequency (min$^{-1}$)")
    # plt.plot(x, np.nanmean(hrepeat, axis=0), linewidth=lw, label="Repeat")
    # plt.plot(x, np.nanmean(hmarch, axis=0), linewidth=lw, label="March")
    # plt.plot(x, np.nanmean(hunetpp, axis=0), linewidth=lw, label="unetpp")
    # plt.plot(x, np.nanmean(hunet3p, axis=0), linewidth=lw, label="unet3p")
    # plt.plot(x, np.nanmean(hunet3p_dsv, axis=0), linewidth=lw, label="unet3p_dsv")
    # plt.plot(x, np.nanmean(htruth, axis=0), linewidth=lw, label="Truth")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(bp_configs.prod_dir + "/figures/hor_psd.png")

def plot_heatmaps():

    rep = [[] for i in range(4)]
    march = [[] for i in range(4)]
    unetpp = [[] for i in range(4)]
    unet3p = [[] for i in range(4)]
    unet3p_dsv = [[] for i in range(4)]

    for file in glob.glob(bp_configs.prod_dir + "/prod_eval/*.npy"):
        print(file)

        # if 'nsa' in file:
        #     continue

        basename = os.path.basename(file)
        month = int(basename[14:16])
        if basename[5:16] in bp_configs.FINAL_PERIODS:
            hits_n_misses = np.load(file, allow_pickle=True)[5]
            print(hits_n_misses)
            for i,item in enumerate(hits_n_misses):
                if i < 4:
                    unet3p_dsv[i%4].append(item)
                elif i < 8:
                    unet3p[i%4].append(item)
                elif i < 12:
                    unetpp[i%4].append(item)
                elif i < 16:
                    continue
                    # unetpp[i%4].append(item)
                elif i < 20:
                    march[i%4].append(item)
                elif i < 24:
                    rep[i%4].append(item)

    print(rep)
    ct_repeat = np.sum(np.asarray(rep), axis=1).reshape((2,2)).T
    ct_march = np.sum(np.asarray(march), axis=1).reshape((2,2)).T
    ct_unetpp = np.sum(np.asarray(unetpp), axis=1).reshape((2,2)).T
    ct_unet3p = np.sum(np.asarray(unet3p), axis=1).reshape((2,2)).T
    ct_unet3p_dsv = np.sum(np.asarray(unet3p_dsv), axis=1).reshape((2,2)).T

    forecast = ["Cloud", "No Cloud"]
    observed = ["Cloud", "No Cloud"]

    res = [ct_repeat, ct_march, ct_unet3p, ct_unetpp, ct_unet3p_dsv]
    titles = ['REP', 'MAR', '++', '3+_1', '3+_5']

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    axes = axes.flatten()
    for k, ax in enumerate(axes):
        if k > 4:
            break
        im = ax.imshow(res[k])
        ax.set_xticks(np.arange(len(forecast)), labels=forecast)
        ax.set_yticks(np.arange(len(observed)), labels=observed)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(forecast)):
            for j in range(len(observed)):
                text = ax.text(j, i, round(res[k][i, j], 1), ha="center", va="center", color="w")
        ax.set_title(titles[k])
        ax.set_xlabel("Observed")
        ax.set_ylabel("Forecast")

    fig.suptitle("Cloud Forecast Contingency Tables")
    fig.tight_layout()
    plt.savefig(bp_configs.prod_dir + "/figures/cloud.png")


def plot_training_curves(path):
    plt.rcParams.update({'font.size': 18})
    df = pd.read_csv(path)

    unetpp_loss = df['ethereal-capybara-14 - epoch/loss']
    unetpp_val_loss = df['ethereal-capybara-14 - epoch/val_loss']

    unet3p_loss = df['sleek-galaxy-12 - epoch/loss']
    unet3p_val_loss = df['sleek-galaxy-12 - epoch/val_loss']

    unet3p_dsv_loss = df['grateful-pyramid-11 - epoch/lambda_1_loss']
    unet3p_dsv_val_loss = df['grateful-pyramid-11 - epoch/val_lambda_1_loss']

    steps = np.arange(len(unetpp_loss))
    fig, ax = plt.subplots(figsize=(15,7))
    plt.title("Model Training Curves")
    plt.ylabel(('MAE Loss (dBZ)'))
    plt.xlabel(('Epoch'))
    plt.plot(steps, unetpp_loss, linewidth=3, color='red', label="Unet++")
    plt.plot(steps, unetpp_val_loss, linewidth=2, color='red', alpha=0.5, linestyle='-')
    plt.plot(steps, unet3p_loss, linewidth=3, color='orange', label="Unet3+")
    plt.plot(steps, unet3p_val_loss, linewidth=2, color='orange', alpha=0.5, linestyle='-')
    plt.plot(steps, unet3p_dsv_loss, linewidth=3, color='blue', label="Unet3+ (DSV)")
    plt.plot(steps, unet3p_dsv_val_loss, linewidth=2, color='blue', alpha=0.5, linestyle='-')

    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/training_curve.pdf')

def plot_summary_vars():
    plt.rcParams.update({'font.size': 14})

    hss_tracker = [[] for i in range(6)]
    mae_tracker = [[] for i in range(6)]
    emd_tracker = [[] for i in range(6)]
    rmse_tracker = [[] for i in range(6)]
    dice_tracker = [[] for i in range(6)]

    for file in glob.glob(bp_configs.prod_dir + "/prod_eval/*.npy"):
        print(file)

        basename = os.path.basename(file)

        # if 'nsa' in basename:
        #     continue
        month = int(basename[14:16])
        data = np.load(file, allow_pickle=True)[2]
        if basename[5:16] in bp_configs.FINAL_PERIODS:
            print(basename[5:16])
        # if month == 1 or month == 2 or month == 3 or month == 4 or month == 10 or month == 11 or month == 12:
            scores = np.load(file, allow_pickle=True)[4]
            maes = np.load(file, allow_pickle=True)[0]
            emds = np.load(file, allow_pickle=True)[1]
            rmses = np.load(file, allow_pickle=True)[2]
            dices = np.load(file, allow_pickle=True)[3]
            for i in range(6):
                hss_tracker[i].append(scores[i])
                rmse_tracker[i].append(rmses[i])
                emd_tracker[i].append(emds[i])
                dice_tracker[i].append(dices[i])
                mae_tracker[i].append(maes[i])

    size = len(hss_tracker[0])

    # SSs = np.nanmean(hss_tracker, axis=1)
    MAEs = np.nanmean(mae_tracker, axis=1)
    print(MAEs)
    RMSEs = np.nanmean(rmse_tracker, axis=1)
    EMDs = np.nanmean(emd_tracker, axis=1).flatten()
    DICEs = np.nanmean(dice_tracker, axis=1)

    MAE_std = np.nanstd(mae_tracker, axis=1) / math.sqrt(size)
    RMSE_std = np.nanstd(rmse_tracker, axis=1) / math.sqrt(size)
    EMD_std = np.nanstd(rmse_tracker, axis=1).flatten() / math.sqrt(size)
    DICE_std = np.nanstd(dice_tracker, axis=1) / math.sqrt(size)

    # models = ['REP', 'MAR', 'U++', '3+', '3+DS']
    models = ['3+_5', '3+_1', '++', 'MAR', 'REP']
    MAEs = [MAEs[0], MAEs[1], MAEs[3], MAEs[4], MAEs[5]]
    DICEs = [DICEs[0], DICEs[1], DICEs[3], DICEs[4], DICEs[5]]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    plt.suptitle('Performance All Years (Winter)')
    ax1.bar(models, MAEs, color='#fe0000', edgecolor='#d70000', linewidth=2, width = 0.66)
    ax1.set_ylabel(('MAE (dBZ)'))
    ax1.set_xlabel('Model')
    ax1.set_ylim((7, 11))
    print(MAEs)

    # ax1.bar(models, RMSEs, yerr=RMSE_std, color='#fe0000', edgecolor='#d70000', linewidth=2, width = 0.66)
    # ax1.set_ylabel(('RMSE (dBZ)'))
    # ax1.set_xlabel('Model')
    # print(RMSEs)
    
    ax2.bar(models, DICEs, color='#4467ff', edgecolor='#3350ca', linewidth=2, width = 0.66)
    ax2.set_ylabel(('EMD (dBZ)'))
    ax2.set_xlabel('Model')

    print(DICEs)

    ax3.bar(models, DICEs,  color='#ffd800', edgecolor='#d8b700', linewidth=2, width = 0.66)
    ax3.set_ylabel(('DICE Score'))
    ax3.set_xlabel('Model')
    ax3.set_ylim((0.3, 0.6))
    print(EMDs)

    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + '/figures/stats.pdf')

def plot_dropout_std():
    eval_files = glob.glob(bp_configs.eval_dir + "/iters/dsv_pred_*.npy")

    all_data = []
    for file in eval_files:
        data = np.load(file, allow_pickle=True)
        all_data.append(data[:,:,:,0])
        
    arr = np.asarray(all_data)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0) / math.sqrt(50)

    for i, mean in enumerate(means):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,12))
        plt.suptitle("Case" + str(i))
        ax1.imshow(mean, vmin=-60, vmax=30, cmap='gist_ncar', interpolation='none')
        ax1.set_title('Avg Reflectivity')
        ax1.invert_yaxis()
        ax2.imshow(stds[i], vmin=0, vmax=3, cmap='Reds', interpolation='none')
        ax2.set_title('STD Refl')
        ax2.invert_yaxis()
        plt.tight_layout()
        plt.savefig(bp_configs.eval_dir + "/iters/figs/case_" + str(i) + ".png")
    
def plot_hists():
    plt.rcParams.update({'font.size': 34})
    # predict_files = glob.glob(bp_configs.prod_dir + "/prod_eval/predictions/*.npy")

    true_arr = []
    true_arr_height = []
    rep_arr = []
    rep_arr_height = []
    mar_arr = []
    mar_arr_height = []
    unet_arr = []
    unet_arr_height = []
    models = ['unet3p_hybrid_50', 'marching', 'repeating'] #'unet3p_hybrid_real_50', unet3p_hybrid_50

    data_paths = bp_utility.path_builder()
    for k, path in enumerate(data_paths):
        if os.path.isfile(bp_configs.data_dir + '/test_set/test_set_' + path + '_kazr.npy'):
            print("Working on", path)

            name = os.path.basename(path)
            basename = os.path.basename(bp_configs.data_dir + '/test_set/test_set_' + path + '_kazr.npy')
            if basename[9:20] in bp_configs.FINAL_PERIODS:
                print(basename[9:20])

                data = np.squeeze(np.load(bp_configs.data_dir + '/test_set/test_set_' + path + '_kazr.npy', allow_pickle=True))
                ds_outer = np.load(bp_configs.data_dir + '/test_set/test_set_' + name + '_kazr.npy', allow_pickle=True)
                ds_truth = np.load(bp_configs.prod_dir + '/prod_eval/predictions/' + name + '_truth.npy', allow_pickle=True)
                
                for i in range(data.shape[0]):
                    if np.max(bp_utility.inv_standardize(ds_outer[i], 'ref', 'kazr')) > 20:
                        continue

                    refls = bp_utility.inv_standardize(ds_truth[i], 'ref', 'kazr').flatten()
                    nan_indices = np.where(refls <= -55)[0]
                    refls = np.delete(refls, nan_indices)
                    heights = np.delete(np.repeat(np.arange(16),128), nan_indices)   
                    true_arr.append(refls)
                    true_arr_height.append(heights)

                    for j, model in enumerate(models):
                        refls = -1
                        ds_mod = np.load(bp_configs.prod_dir + 'prod_eval/predictions/' + name + '_' + model + '.npy', allow_pickle=True)
                        if j > 0:
                            refls = bp_utility.inv_standardize(ds_mod[i], 'ref', 'kazr').flatten()
                        else:
                            invs = []
                            for k in range(bp_configs.N_MC_TESTS):
                                invs.append(bp_utility.inv_standardize(ds_mod[k][i], 'ref', 'kazr'))

                            se = np.squeeze(np.nanstd(np.asarray(invs), axis=0) / math.sqrt(bp_configs.N_MC_TESTS))
                            se_mask = np.where(se > bp_configs.STD_CUTOFF, np.nan, 1)

                            predicted_blind_zone_avg = np.squeeze(np.nanmean(np.asarray(invs), axis=0))
                            predicted_blind_zone_avg = np.multiply(predicted_blind_zone_avg, se_mask)
                            predicted_blind_zone_avg[np.isnan(predicted_blind_zone_avg)] = -60
                            predicted_blind_zone_avg[predicted_blind_zone_avg < -60] = -60
                            predicted_blind_zone_avg[predicted_blind_zone_avg > 30] = 30
                            refls = predicted_blind_zone_avg.flatten()

                        nan_indices = np.where(refls <= -55)[0]
                        refls = np.delete(refls, nan_indices)
                        heights = np.delete(np.repeat(np.arange(16),128), nan_indices)
                        if j == 0:
                            unet_arr.append(refls)
                            unet_arr_height.append(heights)
                        elif j == 1:
                            mar_arr.append(refls)
                            mar_arr_height.append(heights)
                        if j == 2:
                            rep_arr.append(refls)
                            rep_arr_height.append(heights)
                    # break
                    
    unet_refls = [item for sublist in unet_arr for item in sublist]
    unet_heights = [item for sublist in unet_arr_height for item in sublist]
    true_refls = [item for sublist in true_arr for item in sublist]
    true_heights = [item for sublist in true_arr_height for item in sublist]
    mar_refls = [item for sublist in mar_arr for item in sublist]
    mar_heights = [item for sublist in mar_arr_height for item in sublist]
    rep_refls = [item for sublist in rep_arr for item in sublist]
    rep_heights = [item for sublist in rep_arr_height for item in sublist]

    def convert_scale(x):
        return 66.6875 * x + 130

    old_ticks = np.arange(0, 17, 2)  

    # print(len(true_refls), len(true_heights))
    fig, axes = plt.subplots(1, 4, figsize=(30,7), sharey=True)
    axes[0].set_title('KaZR')
    axes[0].set_facecolor('#440154')
    axes[0].set_xlabel('Reflectivity (dBZ)')
    axes[0].set_ylabel('Height (m)')
    h = axes[0].hist2d(true_refls, true_heights, bins=(128,16), cmap=plt.cm.viridis, vmin=0, vmax=20000)#, vmin=0, vmax=75000)#, norm=LogNorm(vmin=2500, vmax=100000))
    new_ticks = [convert_scale(x) for x in old_ticks]
    axes[0].set_yticks(old_ticks)
    axes[0].set_yticklabels([f'{tick:.0f}' for tick in new_ticks]) 
    axes[0].set_xlim((-60, 30))
    # plt.savefig(bp_configs.prod_dir + '/figures/hists/refl_true.png')

    # print(len(rep_refls), len(rep_heights))
    # fig, ax = plt.subplots(figsize=(12,12))
    axes[1].set_title("REP")
    axes[1].set_facecolor('#440154')
    h = axes[1].hist2d(rep_refls, rep_heights, bins=(128,16), cmap=plt.cm.viridis, vmin=0, vmax=20000)#, vmin=0, vmax=75000)#, norm=LogNorm(vmin=2500, vmax=100000))
    axes[1].set_xlabel('Reflectivity (dBZ)')
    # axes[1].set_ylabel('Bin')
    # fig.colorbar(h[3])
    axes[1].set_xlim((-60, 30))
    # plt.savefig(bp_configs.prod_dir + '/figures/hists/refl_repeating.png')

    # print(len(mar_refls), len(mar_heights))
    # fig, ax = plt.subplots(figsize=(12,12))
    axes[2].set_title("MAR")
    axes[2].set_facecolor('#440154')
    axes[2].set_xlabel('Reflectivity (dBZ)')
    # axes[2].set_ylabel('Bin')
    h = axes[2].hist2d(mar_refls, mar_heights, bins=(128,16), cmap=plt.cm.viridis, vmin=0, vmax=20000)#, vmin=0, vmax=75000)#, norm=LogNorm(vmin=2500, vmax=100000))
    # fig.colorbar(h[3])
    axes[2].set_xlim((-60, 30))
    # plt.savefig(bp_configs.prod_dir + '/figures/hists/refl_marching.png')

    # print(len(unet_refls), len(unet_heights))
    # fig, ax = plt.subplots(figsize=(12,12))
    axes[3].set_title("3+_5")
    axes[3].set_facecolor('#440154')
    axes[3].set_xlabel('Reflectivity (dBZ)')
    # axes[3].set_ylabel('Bin')
    h = axes[3].hist2d(unet_refls, unet_heights, bins=(128,16), cmap=plt.cm.viridis, vmin=0, vmax=20000)#, vmin=0, vmax=75000)#, norm=LogNorm(vmin=2500, vmax=100000))
    # fig.colorbar(h[3])
    axes[3].set_xlim((-60, 30))
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + '/figures/hists/full.png')


def plot_performance():
    plt.rcParams.update({'font.size': 26})

    pod_cloud = [0.60, 0.58, 0.88, 0.91, 0.925]
    sr_cloud = [0.97, 0.95, 0.74, 0.75, 0.765]
    cls_cloud = [0.59, 0.57, 0.68, 0.72, 0.72]

    pod_shallow = [0, 0.2, 0.36, 0.22, 0.2425]
    sr_shallow = [0, 0.40, 0.4, 0.54, 0.545]
    cls_shallow = [0, 0.1, 0.23, 0.20, 0.19]

    pod_virga = [0, 0.14, 0.34, 0.37, 0.42]
    sr_virga = [0, 0.26, 0.41, 0.46, 0.43]
    cls_virga = [0, 0.10, 0.23, 0.26, 0.27]

    pods = [pod_cloud, pod_shallow, pod_virga]
    srs = [sr_cloud, sr_shallow, sr_virga]
    clss = [cls_cloud, cls_shallow, cls_virga]
    color_labels = ['3+_5', '3+_1', '++', 'MAR', 'REP']
    symbol_labels = ['cloud', 'shallow', 'virga']
    styles=['o', 's', '^']
    colors=['gold', 'green', 'orange', 'blue', 'red']

    fig, ax = plt.subplots(figsize=(15,13))
    contour_cls = np.outer(np.linspace(0, 1, 100), np.linspace(0, 1, 100)) 
    # plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='black', zorder=2)

    # Plot contour color
    contourf = ax.contourf(np.linspace(0, 1, 100), np.linspace(0, 1, 100), contour_cls, cmap="Blues", levels=15)

    # Draw frequency bias lines
    FBs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for FB in FBs:
        sr_values = np.linspace(0.01, 1, 100)  # avoid division by zero
        pod_values = FB * sr_values
        alpha=0.4
        if FB == 1.0:
            alpha=1
        plt.plot(sr_values, pod_values,  'k--', linewidth=2, alpha=alpha)

    for i in range(3):

        pod = pods[i]
        sr = srs[i]
        cls = clss[i]

        # Scatter plot
        sc = ax.scatter(sr, pod, c=colors, s=600, marker=styles[i], edgecolor='white', linewidth=2, zorder=5)

    # Adding colorbar
    cb = fig.colorbar(contourf)
    cb.set_label('CSI (critical success index)')

    # Set labels
    ax.set_xlabel('Success Ratio (1 - FAR)')
    ax.set_ylabel('Probability of Detection (POD)')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    # Create legend
    legend_elements = [Patch(facecolor=color, label=label) for color, label in zip(colors, color_labels)]
    legend_elements += [Line2D([0], [0], marker=marker, color='black', label=label,
                            markerfacecolor='black', markersize=15, linewidth=0) for marker, label in zip(styles, symbol_labels)]
    ax.legend(loc='upper center', handles=legend_elements, bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=4)
    plt.tight_layout()
    plt.savefig('perf_graph.png')

def find_best_months():
    diffs = []

    for file in glob.glob(bp_configs.prod_dir + "/prod_eval/*.npy"):
        basename = os.path.basename(file)
        site = basename[5:8]
        year = basename[9:13]
        month = int(basename[14:16])
        data = np.load(file, allow_pickle=True)[2]
        # if month == 2 or month == 3 or month == 10 or month == 11:
        diffs.append(data[0] - data[4])
        print(site, year, month, (data[0] - data[4]))

    fig, ax = plt.subplots(figsize=(12,12))
    plt.plot(np.arange(len(diffs)), diffs)
    plt.ylim((-1, 1))
    plt.axhline(0, linestyle='--')
    plt.axhline(np.nanmean(diffs), linestyle='-')
    plt.show()
    print(diffs)

    
# find_best_months()

# # Monthly comparisons
build_monthly_intercomparisons('MAE', '(dBZ)', 0)
# build_monthly_intercomparisons('EMD', '(dBZ)', 1)
build_monthly_intercomparisons('DICE', '(SS)', 3)
# build_monthly_intercomparisons('RMSE','(dBZ)', 2)

# # PSD Stuff
plot_psds()

# # Heatmap stuff
# plot_heatmaps()

# # Bars
# plot_summary_vars()

# # Performance plot
# plot_performance()

# # Histograms
# plot_hists()

# # Curves
# plot_training_curves('/Users/fraserking/Desktop/holders/wandb_export_2023-03-28T11_00_24.718-04_00.csv')

print("All done!")