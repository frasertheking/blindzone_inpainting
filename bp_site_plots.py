#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: plot various skill metrics and histograms for the inpainted test regions
"""

import os, math
import bp_configs
import bp_utility
import numpy as np
import glob
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from seaborn_qqplot import pplot
from sre_constants import OP_LOCALE_IGNORE
plt.rcParams.update({'font.size': 22})

SITES_TO_EXAMINE = ['nsa', 'oli']
vars = ['ref', 't', 'q', 'u', 'v']

def path_builder():
    data_paths = []
    for site in SITES_TO_EXAMINE:
        for year in bp_configs.YEARS_TO_EXAMINE:
            for month in bp_configs.MONTHS_TO_EXAMINE:
                data_paths.append(site + '_' + year + '_' + month)
    return data_paths

data_paths = path_builder()

def plot_histograms_for_vars():
    for month in bp_configs.MONTHS_TO_EXAMINE:
        print("ON MONTH", month)
        refl_set = []
        t_set = []
        q_set = []
        u_set = []
        v_set = []
        height_set = []
        height_set2 = []
        height_set4 = []
        height_set5 = []
        height_set3 = []
        count = 0
        for i, path in enumerate(data_paths):
            basename = os.path.basename(path)
            if month in basename[-2:]:
                print(month, basename)
                data_file = bp_configs.data_dir + 'preprocessed/' + path + '_kazr.npy'
                if os.path.isfile(data_file):
                    print("Working on", data_file)
                    for j in range(1):#bp_configs.CHANNELS):
                        data = np.load(data_file)[:,:,j] # get refl
                        if j == 0:
                            refls = bp_utility.inv_standardize(data,vars[j],'kazr').flatten()
                            # r_indices = list(np.arange(len(refls)))
                            # r_indices = random.sample(r_indices, int(len(r_indices)*0.01))
                            # refls = [refls[i] for i in r_indices]
                            nan_indices = np.where(refls == -60)[0]
                            refls = np.delete(refls, nan_indices)
                            heights = np.delete(np.repeat(np.arange(128),data.T.shape[0]), nan_indices)   
                            refl_set.append(refls)
                            height_set.append(heights)
                        elif j == 1:
                            t_vals = bp_utility.inv_standardize(data,vars[j],'kazr').flatten()
                            t_indices = list(np.arange(len(t_vals)))
                            t_indices = random.sample(t_indices, int(len(t_indices)*0.01))
                            t_vals = [t_vals[i] for i in t_indices]
                            t_set.append(t_vals)
                            heights = np.repeat(np.arange(128),data.T.shape[0])
                            heights = [heights[i] for i in t_indices]
                            height_set2.append(heights)
                        elif j == 2:
                            q_vals = bp_utility.inv_standardize(data,vars[j],'kazr').flatten()
                            q_indices = list(np.arange(len(q_vals)))
                            q_indices = random.sample(q_indices, int(len(q_indices)*0.01))
                            q_vals = [q_vals[i] for i in q_indices]
                            q_set.append(q_vals)
                            heights = np.repeat(np.arange(128),data.T.shape[0])
                            heights = [heights[i] for i in q_indices]
                            height_set3.append(heights)
                        elif j == 3:
                            vals = bp_utility.inv_standardize(data,vars[j],'kazr').flatten()
                            indices = list(np.arange(len(vals)))
                            indices = random.sample(indices, int(len(indices)*0.01))
                            vals = [vals[i] for i in indices]
                            u_set.append(vals)
                            heights = np.repeat(np.arange(128),data.T.shape[0])
                            heights = [heights[i] for i in indices]
                            height_set4.append(heights)
                        elif j == 4:
                            vals = bp_utility.inv_standardize(data,vars[j],'kazr').flatten()
                            indices = list(np.arange(len(vals)))
                            indices = random.sample(indices, int(len(indices)*0.01))
                            vals = [vals[i] for i in indices]
                            v_set.append(vals)
                            heights = np.repeat(np.arange(128),data.T.shape[0])
                            heights = [heights[i] for i in indices]
                            height_set5.append(heights)

                    # if count > 0:
                    #     break
                    # count += 1
                    # break

        refl_set = np.concatenate(refl_set).ravel()
        # t_set = np.concatenate(t_set).ravel()
        # q_set = np.concatenate(q_set).ravel()
        # u_set = np.concatenate(u_set).ravel()
        # v_set = np.concatenate(v_set).ravel()
        height_set = np.concatenate(height_set).ravel()
        # height_set2 = np.concatenate(height_set2).ravel()
        # height_set3 = np.concatenate(height_set3).ravel()
        # height_set4 = np.concatenate(height_set4).ravel()
        # height_set5 = np.concatenate(height_set5).ravel()

        refl_indices = list(np.arange(len(refl_set)))
        print("Randomly subsampling", int(len(refl_indices)*0.1), len(refl_indices))
        refl_indices = random.sample(refl_indices, int(len(refl_indices)*0.1))
        refl_set = [refl_set[i] for i in refl_indices]
        height_set = [height_set[i] for i in refl_indices]

        # print(len(refl_set), len(t_set))
        # t_indices = list(np.arange(len(t_set)))
        # print("Randomly subsampling", int(len(t_indices)*0.1), len(t_indices))
        # t_indices = random.sample(t_indices, int(len(t_indices)*0.1))
        # t_set = [t_set[i] for i in t_indices]
        # q_set = [q_set[i] for i in t_indices]
        # u_set = [u_set[i] for i in t_indices]
        # v_set = [v_set[i] for i in t_indices]
        # height_set2 = [height_set2[i] for i in t_indices]
        # height_set3 = [height_set3[i] for i in t_indices]
        # height_set4 = [height_set4[i] for i in t_indices]
        # height_set5 = [height_set5[i] for i in t_indices]

        
        fig, ax = plt.subplots(figsize=(13,10))
        h = plt.hist2d(refl_set, height_set, bins=(64, 64), cmap=plt.cm.Reds, vmin=0, vmax=20000)
        plt.xlim((-60, 30))
        plt.title(month + " - Reflectivity")
        plt.xlabel("Reflectivity (dBZ)")
        plt.ylabel("Bin")
        fig.colorbar(h[3], ax=ax)
        plt.tight_layout()
        plt.savefig('figures/hist_' + month + '_refl.png')

        # fig, ax = plt.subplots(figsize=(13,10))
        # h = plt.hist2d(t_set, height_set2, bins=(64, 64), cmap=plt.cm.Blues)
        # plt.xlim((200, 300))
        # plt.title(site + " - ERA5 Temperature")
        # plt.xlabel("Temperature (K)")
        # plt.ylabel("Bin")
        # fig.colorbar(h[3], ax=ax)
        # plt.tight_layout()
        # plt.savefig('figures/site_' + site + '_temp.png')

        # fig, ax = plt.subplots(figsize=(13,10))
        # h = plt.hist2d(u_set, height_set4, bins=(64, 64), cmap=plt.cm.Oranges)
        # plt.xlim((-25, 50))
        # plt.title(site + " - ERA5 Wind (U-component)")
        # plt.xlabel("U-component of Wind Speed (m/$s^2$)")
        # plt.ylabel("Bin")
        # fig.colorbar(h[3], ax=ax)
        # plt.tight_layout()
        # plt.savefig('figures/site_' + site + '_u.png')

        # fig, ax = plt.subplots(figsize=(13,10))
        # h = plt.hist2d(v_set, height_set5, bins=(64, 64), cmap=plt.cm.Greens)
        # plt.xlim((-25, 35))
        # plt.title(site + " - ERA5 Wind (V-component)")
        # plt.xlabel("V-component of Wind Speed (m/$s^2$)")
        # plt.ylabel("Bin")
        # fig.colorbar(h[3], ax=ax)
        # plt.tight_layout()
        # plt.savefig('figures/site_' + site + '_v.png')

        # fig, ax = plt.subplots(figsize=(13,10))
        # h = plt.hist2d(q_set, height_set3, bins=(64, 64), cmap=plt.cm.Purples, norm=LogNorm(vmin=75, vmax=50000))
        # plt.xlim((0, 0.0065))
        # plt.title(site + " - ERA5 Specific Humidity")
        # plt.xlabel("Specific Humidity (kg/kg)")
        # plt.ylabel("Bin")
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # fig.colorbar(h[3], ax=ax)
        # plt.tight_layout()
        # plt.savefig('figures/site_' + site + '_sh.png')


def plot_summary_vars():
    for site in SITES_TO_EXAMINE:
        print("ON SITE", site)
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        variables = ['Reflectivity', 'Temperature', 'Specific Humidity', 'Wind Speed (U-Component)', 'Wind Speed (V-Component)']
        shorts = ['ref', 't', 'q', 'u', 'v']
        units = ['dBZ', 'K', 'kg/kg', 'm/s$^2$', 'm/s$^2$']
        cmaps = ['Reds', 'Blues', 'Purples', 'Oranges', 'Greens']

        for j in range(bp_configs.CHANNELS):
            month_temps = [[] for i in range(12)]
            month_temps_heatmap = [[] for i in range(12)]
            if j == 0:
                continue
            for i, path in enumerate(data_paths):
                basename = os.path.basename(path)
                if site in basename:
                    data_file = bp_configs.data_dir + 'preprocessed/' + path + '_kazr.npy'
                    if os.path.isfile(data_file):
                        print("Working on", data_file)
                        data = np.load(data_file)[:,:,j]
                        data = np.asarray(data, dtype=np.float64)                        
                        data = bp_utility.inv_standardize(data, shorts[j], 'kazr')
                        month_temps_heatmap[int(basename[-2:])-1].append(np.nanmean(data, axis=1))
                        month_temps[int(basename[-2:])-1].append(np.nanmean(data))

            final_temps = []
            final_temp_stds = []
            for month in month_temps:
                final_temps.append(np.mean(month))
                final_temp_stds.append(np.std(month))

            final_temp_stds = np.asarray(final_temp_stds)
            final_temps_heatmap = []
            for month in month_temps_heatmap:
                final_temps_heatmap.append(np.nanmean(month, axis=0))

            fig, ax = plt.subplots(figsize=(15,12))
            plt.title('ERA5 ' + variables[j] + ' (All Years - ' + site + ')')
            h = plt.imshow(np.asarray(final_temps_heatmap, dtype=np.float64).T, aspect='auto', interpolation='none', cmap=cmaps[j])
            ax.set_ylabel('Height')
            ax2 = ax.twinx()
            # ax2.set_ylabel('Temperature (K)')
            ax.set_xlabel('Month')
            ax.invert_yaxis()
            ax2.plot(months, final_temps, linewidth=3, color='black')
            ax2.scatter(months, final_temps, s=50, color='black')
            ax2.fill_between(months, (final_temps + 2*final_temp_stds), (final_temps - 2*final_temp_stds), alpha=0.1, color='black')
            fig.colorbar(h, label=variables[j] + ' (' + units[j] + ')', orientation="vertical", pad=0.1)
            plt.tight_layout()
            plt.savefig(bp_configs.prod_dir + '/figures/' + shorts[j] + '_' + site + '_summary.png') 

def plot_cloud_stats():
    plt.rcParams.update({'font.size': 35})

    actual_bases = []
    pred_bases = []
    pred_bases2 = []
    pred_bases3 = []
    mar_bases = []
    rep_bases = []

    month_bases = [[] for i in range(12)]
    month_tops = [[] for i in range(12)]
    month_cloud_fractions = [[] for i in range(12)]
    month_layer_counts = [[] for i in range(12)]

    for file in glob.glob(bp_configs.prod_dir + "/prod_eval/*.npy"):
        basename = os.path.basename(file)
        if basename[5:16] in bp_configs.FINAL_PERIODS:
            print(basename[5:16])
            basename = os.path.basename(file)
            month = int(basename[14:16])

            cloud_features = np.load(file, allow_pickle=True)[10] #11
            cloud_base = np.load(file, allow_pickle=True)[11] #11
            month_tops[month-1].append(cloud_features[0])
            month_bases[month-1].append(cloud_features[1])
            month_cloud_fractions[month-1].append(cloud_features[2])
            month_layer_counts[month-1].append(cloud_features[3])

            actual_bases.append(cloud_features[1])
            pred_bases.append(np.nanmean(cloud_base[0]))
            pred_bases2.append(np.nanmean(cloud_base[1]))
            pred_bases3.append(np.nanmean(cloud_base[3]))
            mar_bases.append(np.nanmean(cloud_base[4]))
            rep_bases.append(np.nanmean(cloud_base[5]))

    df = pd.DataFrame(data={'truth': actual_bases, '3p_hy': pred_bases, '3p_3': pred_bases2, '3p_1': pred_bases3, 'mar': mar_bases, 'rep': rep_bases})

    # calculate quantiles
    quantiles_num = 100
    quantiles_pred = np.percentile(pred_bases, np.linspace(0, 100, quantiles_num))
    quantiles_pred2 = np.percentile(pred_bases2, np.linspace(0, 100, quantiles_num))
    quantiles_pred3 = np.percentile(pred_bases3, np.linspace(0, 100, quantiles_num))
    quantiles_mar = np.percentile(mar_bases, np.linspace(0, 100, quantiles_num))
    quantiles_rep = np.percentile(rep_bases, np.linspace(0, 100, quantiles_num))
    quantiles_truth = np.percentile(actual_bases, np.linspace(0, 100, quantiles_num))

    # pp_x = sm.ProbPlot(quantiles_pred2)
    # pp_y = sm.ProbPlot(quantiles_truth)
    # qqplot_2samples(pp_x, pp_y)
    # plt.show()

    size=350
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    plt.title("Lowest Reflectivity Bin Distributions") 
    plt.plot([np.min((quantiles_pred.min(),quantiles_truth.min())), np.max((quantiles_pred.max(),quantiles_truth.max()))], 
            [np.min((quantiles_pred.min(),quantiles_truth.min())), np.max((quantiles_pred.max(),quantiles_truth.max()))], color='black', linestyle='--', linewidth=6)
    
    plt.scatter(quantiles_pred3, quantiles_truth, s=size, facecolors='none', edgecolors='orange', linewidths=3, label="++", zorder=100)
    plt.scatter(quantiles_pred, quantiles_truth, s=size, facecolors='none', edgecolors='red', linewidths=3, label="3+_5", zorder=100)
    plt.scatter(quantiles_pred2, quantiles_truth, s=size, facecolors='none', edgecolors='blue', linewidths=3, label="3+_1", zorder=100)
    plt.scatter(quantiles_mar, quantiles_truth, s=size, facecolors='none', edgecolors='green', linewidths=3, label="MAR", zorder=100)
    plt.scatter(quantiles_rep, quantiles_truth, s=size, facecolors='none', edgecolors='gold', linewidths=3, label="REP", zorder=100)
    plt.xlabel('Predicted Lowest Bin')
    plt.ylabel('KaZR Lowest Bin')
    plt.xlim((0, 40))
    plt.ylim((0, 40))
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + '/figures/cloud_base_rep.png')

    # fig, ax = plt.subplots(1, 1, figsize=(10,10))
    # plt.title("Cloud Base Distributions")
    # vals = pplot(data=df, x="3p_hy", y="truth", height=10, kind='qq', display_kws={"identity":False})
    # print(vals)
    # # pplot(data=df, x="3p_3", y="truth", height=10, kind='qq', display_kws={"identity":False})
    # # pplot(data=df, x="3p_1", y="truth", height=10, kind='qq', display_kws={"identity":False})
    # # pplot(data=df, x="mar", y="truth", height=10, kind='qq', display_kws={"identity":False})
    # #pplot(data=df, x="rep", y="truth", height=5, kind='qq', display_kws={"identity":True, "markersize":200})
    # # plt.plot([0, 0], [50, 50], linestyle='--', color='black', linewidth=2)
    # plt.xlabel('Predicted Cloud Base (bin)')
    # plt.ylabel('True Cloud Base (bin)')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=6)
    # plt.xlim((0, 30))
    # plt.ylim((0, 30))
    # plt.tight_layout()
    # plt.savefig(bp_configs.prod_dir + '/figures/cloud_base_rep.png')

    fig, ax = plt.subplots(1, 1, figsize=(22,15))
    plt.title("Cloud Base Distributions")
    plt.axvline(np.nanmean(df.truth), linewidth=8, alpha=1, color='black', linestyle='--')
    plt.axvline(np.nanmean(df['3p_hy']), linewidth=4, alpha=1, color='red', linestyle='--')
    plt.axvline(np.nanmean(df['3p_3']), linewidth=4, alpha=1, color='blue', linestyle='--')
    plt.axvline(np.nanmean(df['3p_1']), linewidth=4, alpha=1, color='orange', linestyle='--')
    plt.axvline(np.nanmean(df.mar), linewidth=4, alpha=1, color='green', linestyle='--')
    plt.axvline(np.nanmean(df.rep), linewidth=4, alpha=1, color='gold', linestyle='--')
    sns.kdeplot(data=df, x="truth", color='black', linewidth=12, label='Truth')
    sns.kdeplot(data=df, x="3p_hy", color='red', linewidth=8, label='3+_5')
    sns.kdeplot(data=df, x="3p_3", color='blue', linewidth=8, label='3+_1')
    sns.kdeplot(data=df, x="3p_1", color='orange', linewidth=8, label='++')
    sns.kdeplot(data=df, x="mar", color='green', linewidth=8, label='MAR')
    sns.kdeplot(data=df, x="rep", color='gold', linewidth=8, label='REP')
    plt.xlabel('Cloud Base (bin)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=6)
    plt.xlim((0, 80))
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + '/figures/cloud_base_summary.png')

    month_features = [month_bases, month_tops, month_cloud_fractions, month_layer_counts]
    month_feature_names = ['Bases', 'Tops', 'Cloud Fraction', 'Layers']
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    colors=['red', 'blue', 'orange', 'green']
    for j in range(4):
        fig, ax = plt.subplots(1, figsize=(12,6))
        plt.title(month_feature_names[j])
        means = []
        stds = []
        counts= []
        for i in range(12):
            means.append(np.nanmean(month_features[j][i]))
            stds.append(np.nanstd(month_features[j][i]))
            counts.append(len(month_features[j][i]))
        ax.bar(months, means, yerr=1*stds[i]/math.sqrt(counts[i]), color=colors[j])
        plt.savefig(bp_configs.prod_dir + '/figures/cloud_features/' + month_feature_names[j] + '.png')


        # sys.exit()

def plot_data_availability():
    nsa_data = []
    oli_data = []
    for path in data_paths:
        print(path)
        data_file = bp_configs.data_dir + 'preprocessed/' + path + '_kazr.npy'
        if os.path.isfile(data_file):
            if 'nsa' in path:
                nsa_data.append(2)
            else:
                oli_data.append(1)
        else:
            if 'nsa' in path:
                nsa_data.append(np.nan)
            else:
                oli_data.append(np.nan)

    fig, ax = plt.subplots(figsize=(16,8))
    plt.scatter(np.arange(len(nsa_data)), nsa_data, s=300, marker="s", c='#2d88f8')
    plt.scatter(np.arange(len(oli_data)), oli_data, s=300, marker="s", c='#ea4242')
    plt.ylim((0, 3))
    plt.yticklabels
    plt.show()

def plot_temperature_data():
    plt.rcParams.update({'font.size': 28})
    all_data_nsa = []
    all_data_oli = []
    for path in data_paths:
        data_file = bp_configs.data_dir + '/preprocessed_met/' + path + '_temp.npy'
        site = path[:3]
        year = path[4:8]
        month = path[-2:]
        
        if os.path.isfile(data_file):
            data = np.load(data_file)
            data[data<-50]=np.nan
            data[data>50]=np.nan
            max_val = np.nanmax(data)

            if max_val <= 2:
                print(path)

            if 'nsa' in path:
                all_data_nsa.append(np.nanmax(data))
            else:
                all_data_oli.append(np.nanmax(data))
        else:
            if 'nsa' in path:
                all_data_nsa.append(np.nan)
            else:
                all_data_oli.append(np.nan)

    dates = pd.date_range('2012-01', periods=72, freq='M')
    fig, ax = plt.subplots(figsize=(16,8))
    # plt.title("Max Monthly Surface Temperatures")
    plt.plot(dates, all_data_nsa, linewidth=6, color='#2d88f8', label='NSA')
    plt.scatter(dates, all_data_nsa, s=125, color='#2e76ce', zorder=1000)
    plt.axhline(2, linestyle='--', linewidth=4, color='black')
    plt.plot(dates, all_data_oli, linewidth=6, color='#ea4242', label="OLI")
    plt.scatter(dates, all_data_oli, s=125, color='#c92f2f', zorder=1000)
    plt.ylabel("Max. Monthyl Temp. (deg C)")
    plt.xlabel("Month")
    plt.ylim((-20, 30))
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + '/figures/met_monthly_temps.png')

plot_temperature_data()
# plot_data_availability()
# plot_cloud_stats()
# plot_summary_vars()
# plot_histograms_for_vars()