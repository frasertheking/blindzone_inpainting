#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: perform the drop-channel marginal importance calculations
"""

import os, gc, random, copy
import itertools
import bp_models
import bp_configs
import bp_utility
import scipy
import wandb
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import History
from matplotlib.lines import Line2D

number_of_channels = -1
DOWNFILL_SAMPLE_INDS = -1

def calc_MAE(y_pred, y_true):
    return np.nanmean(np.abs(y_pred - y_true))

# Redefined for custom tests here..
def downfill_batch(x, data = None, sample_inds=DOWNFILL_SAMPLE_INDS, cut_range=bp_configs.DOWNFILL_CUT_RANGE, buf_range=bp_configs.BUF_RANGE):
    BS = x.shape[0]
    NT = x.shape[2]
    if not data is None:
        for i in range(BS):
            idx = np.random.choice(sample_inds)
            x[i,:,:,:number_of_channels]  = np.copy(data[:,idx:idx+NT,:])
            if np.random.choice([True, False]):
                x[i,:,:,:] = np.flip(x[i,:,:,:],axis=1)
        
    FS = bp_configs.min_weather_size
    for i in range(BS):
        if len(cut_range)==2:
            mask = np.float16(x[i,cut_range[0]:cut_range[1]+FS,:,0]>-0.5)
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

def blind_MAE(y_true,y_pred):
    filt = y_true[:,:,:,number_of_channels]
    y_true = y_true[:,:,:,:1]
    y_pred = y_pred[:,:,:,:1]
    mae = tf.math.abs(y_true-y_pred)        
    filt = tf.tile(tf.expand_dims(filt, axis=3), [1, 1, 1, number_of_channels])

    weighted_mae = tf.reduce_mean(mae*filt)/tf.reduce_mean(filt)
    return weighted_mae

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


def performSHAP():
    # Build paths
    data_paths = bp_utility.path_builder()

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

    comb_test_set, _, test_breaks = data_combiner(data_paths, 'test_set', axis=0)
    print("\nTest sets loaded", comb_test_set.shape)

    # Samples
    # comb_samples, _, _ = data_combiner(data_paths, 'samples', axis=0)
    # print("\nSamples loaded", comb_samples.shape)

    # Define a function to train and evaluate the model with different input combinations
    def train_evaluate(inputs, count):
        global number_of_channels

        # inputs = ('radar', 'humidity', 'u_wind', 'v_wind')

        print("WORKING ON INPUTS", inputs)

        # Start a run
        wandb.init(
            # set the wandb project where this run will be logged
            project="bp_shap_find_enemy",
        )

        # input_tensors = []
        input_channels = []
        test_channels = []
        sample_channels = []

        channel_size = len(inputs)
        number_of_channels = channel_size
        print("channel size", channel_size)
        batch = downfill_batch

        if 'radar' in inputs:
            input_channels.append(comb_preprocessed[:,:,0])
            test_channels.append(comb_test_set[:,:,:,0])
            # sample_channels.append(comb_samples[:,:,:,0])
        if 'temp' in inputs:
            input_channels.append(comb_preprocessed[:,:,1])
            test_channels.append(comb_test_set[:,:,:,1])
            # sample_channels.append(comb_samples[:,:,:,1])
        if 'humidity' in inputs:
            input_channels.append(comb_preprocessed[:,:,2])
            test_channels.append(comb_test_set[:,:,:,2])
            # sample_channels.append(comb_samples[:,:,:,2])
        if 'u_wind' in inputs:
            input_channels.append(comb_preprocessed[:,:,3])
            test_channels.append(comb_test_set[:,:,:,3])
            # sample_channels.append(comb_samples[:,:,:,3])
        if 'v_wind' in inputs:
            input_channels.append(comb_preprocessed[:,:,4])
            test_channels.append(comb_test_set[:,:,:,4])
            # sample_channels.append(comb_samples[:,:,:,4])

        # sample_channels.append(comb_samples[:,:,:,-1])

        config_defaults = bp_configs.config_defaults
        input_channels = np.asarray(input_channels).T.swapaxes(0, 1)
        test_channels = np.transpose(np.asarray(test_channels), (1, 2, 3, 0))
        # sample_channels = np.transpose(np.asarray(sample_channels), (1, 2, 3, 0))

        #prep the inputs for the CNNs:
        sz = bp_configs.SIZE['downfill'][1]
        dsz = 16
        mask = np.zeros((sz,sz))
        mask[:dsz,:] = 1.0
        buf_size = 8
        buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
        mask[dsz:dsz+buf_size,:] = buf[:,np.newaxis]
        mask = mask[:,:,np.newaxis]
        tmp_test_data = []
        test_channel2 = copy.deepcopy(test_channels)
        for i in range(len(test_channel2)):
            sample = test_channel2[i]        
            for j in range(channel_size):
                sample[:dsz,:,j] = -1.0
            tmp_test_data.append(np.concatenate((sample,mask),axis=2))
        test_data = tmp_test_data

        print(np.asarray(test_data).shape)
        # print(sample_channels.shape)

        print("INPUT CHANNEL SIZE", input_channels.shape)

        # for sample in sample_channels:
        #     for channel in range(number_of_channels+1):
        #         plt.imshow(sample[:,:,channel])
        #         plt.title(str(channel))
        #         plt.show()

        x = np.zeros((250*config_defaults['batch_size'],*bp_configs.SIZE['downfill'],channel_size+1),dtype='float16')
        history = History()

        cnn = bp_models.unet3plus((*bp_configs.SIZE['downfill'],channel_size+1), channel_size, \
                                config=config_defaults, depth=config_defaults['depth'], training=True, clm=False)
        cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config_defaults['lr']), loss=blind_MAE)

        for epoch in range(15):
            gc.collect()
            print('EPOCH ' + str(epoch) + ':')

            batch(x, input_channels, sample_inds = DOWNFILL_SAMPLE_INDS)
            print("L1: Epoch batch size", x.shape)
            cnn.fit(x, x, batch_size=config_defaults['batch_size'], verbose=1, validation_split=0.05, callbacks=[history, WandbMetricsLogger(log_freq=5)])
        
        save_string = ""
        for input in inputs:
            save_string += input + "_"

        cnn.save(bp_configs.prod_dir + '/extras/enemy/' + save_string)
        wandb.finish()
        return -1

    # Define the input channels
    input_channels = ['temp', 'humidity', 'u_wind', 'v_wind']

    # Create all combinations of input channels
    channel_combinations = []
    for r in range(1, len(input_channels) + 1):
        channel_combinations.extend(list(itertools.combinations(input_channels, r)))

    channel_combinations = [('radar',) + combination for combination in channel_combinations][::-1]

    # channel_combinations = [('radar', 'humidity', 'temp', 'u_wind', 'v_wind')]

    # channel_combinations = [('radar',)]
    # Train and evaluate the model for each combination
    results = {}
    count = 0
    for combination in channel_combinations:
        print(f"Training model with inputs: {', '.join(combination)}")
        maes = train_evaluate(combination, count)
        # print("maes", np.nanmean(maes))
        # np.save(bp_configs.prod_dir + '/extras/enemy/combo_maes_' +  str(count) + '.npy', maes)
        # np.save(bp_configs.prod_dir + '/extras/enemy/month_idx_' +  str(count) + '.npy', test_breaks)


        # print(f"Validation MAE: {mae}\n")
        # results[combination] = mae
        # df = pd.DataFrame(data={'combination': combination, 'mae': mae})
        # df.to_csv(bp_configs.prod_dir + '/extras/combo_' + str(count) + '.csv')
        count += 1

    # Print the results
    # for combination, val_mae in results.items():
    #     print(f"Inputs: {', '.join(combination)}, Validation MAE: {val_mae}")

    
def getSHAPValues():
    def calculate_shap_values(cases, results):
        # Convert results to floats and create a dictionary
        results_dict = {case: float(result) for case, result in zip(cases, results)}

        # Define a function to compute the marginal contribution of adding an input channel
        def marginal_contribution(channel, combination, results):
            with_channel = tuple(sorted(combination + (channel,)))
            without_channel = combination
            return results[with_channel] - results.get(without_channel, 0)

        # Calculate the Shapley values
        shapley_values = {}
        input_channels = ['temp', 'humidity', 'u_wind', 'v_wind']
        num_channels = len(input_channels) + 1  # Add 1 to account for the 'radar' channel
        for channel in input_channels:
            shapley_value = 0
            print("\nChannel", channel)
            for r in range(1, len(input_channels) + 1):
                print("Range", r)
                combinations = list(itertools.combinations([ch for ch in input_channels if ch != channel], r))
                print("Combinations", r)
                print()
                for combination in combinations:
                    marginal_contrib = marginal_contribution(channel, combination, results_dict)
                    weight = 1 / (num_channels * scipy.special.comb(num_channels - 1, r))
                    shapley_value += weight * marginal_contrib
                    print("Combo", combination, marginal_contrib, weight, shapley_value, num_channels, scipy.special.comb(num_channels - 1, r))
            shapley_values[channel] = shapley_value / num_channels

        return shapley_values

    cases = [('radar'), ('temp'), ('humidity'), ('u_wind'), \
             ('v_wind'), ('humidity', 'temp'), ('temp', 'u_wind'), \
             ('temp', 'v_wind'), ('humidity', 'u_wind'), \
             ('humidity', 'v_wind'), ('u_wind', 'v_wind'), \
             ('humidity', 'temp', 'u_wind'), ('humidity', 'temp', 'v_wind'), \
             ('temp', 'u_wind', 'v_wind'), ('humidity', 'u_wind', 'v_wind'), \
             ('humidity', 'temp', 'u_wind', 'v_wind')]

    results = [1.196822643,0.951139688,1.113204837,1.158886671,1.159593701,
               1.057475924,0.934354067,0.982720613,0.916070998,1.035811901,
               0.981895149,0.852088392,0.961472154,0.865587234,0.934009492,0.824200068]

    shapley_values = calculate_shap_values(cases, results)

    for channel, shapley_value in shapley_values.items():
        print()
        print(channel)
        print(shapley_value)

def plotSHAP():
    plt.rcParams.update({'font.size': 28})
    # Assuming you already have the Shapley values calculated
    shapley_values = {
        "t": 0.026949876680000002,
        "q": 0.028670384630000007,
        "u": 0.04078622018,
        "v": 0.04431723607666667,
    }

    labels = ['r', 't', 'q', 'u', 'v', 'tq', 'tu', 'tv', 'qu', 'qv', 'uv', 'tqu', 'tqv', 'tuv', 'quv', 'tquv']

    results = [1.196822643,0.951139688,1.113204837,1.158886671,1.159593701,
               1.057475924,0.934354067,0.982720613,0.916070998,1.035811901,
               0.981895149,0.852088392,0.961472154,0.865587234,0.934009492,0.824200068]
    df = pd.read_csv(bp_configs.prod_dir + '/extras/epoch_loss.csv', header=1).T
    df = df.reset_index()  # make sure indexes pair with number of rows

    colors = ['black', '#1D3557', '#1D3557', '#1D3557', '#1D3557', \
               '#457B9D', '#457B9D', '#457B9D', '#457B9D', '#457B9D', '#457B9D', \
                 '#F7A6A4', '#F7A6A4', '#F7A6A4', '#F7A6A4', '#E63946']
    colors_abrv = ['black', '#1D3557', '#457B9D', '#F7A6A4', '#E63946']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors_abrv]
    legend_labs = ['KaZR', '2chan', '3chan', '4chan', '5chan']

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(30,10))
    for index, row in df.iterrows():
        if index == 0:
            continue
        if index == 1:
            ax.plot(np.arange(0, 500, 50), np.asarray(row.values, dtype=np.float64)[:-1], color=colors[index-1], linewidth=3, linestyle='--', alpha=1)
        else:
            ax.plot(np.arange(0, 500, 50), np.asarray(row.values, dtype=np.float64)[:-1], color=colors[index-1], linewidth=3, alpha=0.9)
    ax.legend(lines, legend_labs)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (MAE)')
    ax.set_title('Training Loss')
    ax2.bar(shapley_values.keys(), shapley_values.values(), color='#4467ff', edgecolor='#3350ca', linewidth=3)
    ax2.set_xlabel('Input Channels')
    ax2.set_ylabel('Std. Marginal Contribution')
    ax2.set_title('Drop Channel Importance')
    ax1.bar(labels, [float(i) for i in results], color='#4467ff', edgecolor='#3350ca', linewidth=3)
    ax1.set_xticks(np.arange(len(labels)), labels, rotation=45, ha='center')
    ax1.set_xlabel('Input Channel Combinations')
    ax1.set_ylabel('Loss (MAE)')
    ax1.set_ylim((0.75, 1.25))
    ax1.set_title('Final Validation Loss')
    plt.tight_layout()
    plt.savefig(bp_configs.prod_dir + '/figures/shap_info.png')

def shapCheckSummer(inputs):
    data_paths = bp_utility.path_builder()
    # comb_test_set, _, test_breaks = data_combiner(data_paths, 'test_set', axis=0)
    # print("\nTest sets loaded", comb_test_set.shape)

    in_string = ""
    for input in inputs:
        in_string += input + "_"

    final_maes = []

    for i, path in enumerate(data_paths):
        if os.path.isfile(bp_configs.data_dir + 'test_set/test_set_' + path + '_kazr.npy'):
            print("\n\nTesting on", path)
            comb_test_set = np.load(bp_configs.data_dir + 'test_set/test_set_' + path + '_kazr.npy', allow_pickle=True)
            test_channels = []

            channel_size = len(inputs)
            number_of_channels = channel_size
            print("channel size", channel_size)
            batch = downfill_batch

            if 'radar' in inputs:
                test_channels.append(comb_test_set[:,:,:,0])
            if 'temp' in inputs:
                test_channels.append(comb_test_set[:,:,:,1])
            if 'humidity' in inputs:
                test_channels.append(comb_test_set[:,:,:,2])
            if 'u_wind' in inputs:
                test_channels.append(comb_test_set[:,:,:,3])
            if 'v_wind' in inputs:
                test_channels.append(comb_test_set[:,:,:,4])

            config_defaults = bp_configs.config_defaults
            test_channels = np.transpose(np.asarray(test_channels), (1, 2, 3, 0))

            #prep the inputs for the CNNs:
            sz = bp_configs.SIZE['downfill'][1]
            dsz = 16
            mask = np.zeros((sz,sz))
            mask[:dsz,:] = 1.0
            buf_size = 8
            buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
            mask[dsz:dsz+buf_size,:] = buf[:,np.newaxis]
            mask = mask[:,:,np.newaxis]
            tmp_test_data = []
            test_channel2 = copy.deepcopy(test_channels)
            for i in range(len(test_channel2)):
                sample = test_channel2[i]        
                for j in range(channel_size):
                    sample[:dsz,:,j] = -1.0
                tmp_test_data.append(np.concatenate((sample,mask),axis=2))
            test_data = tmp_test_data

            unet3p_dsv = bp_models.unet3plus((*bp_configs.SIZE['downfill'],number_of_channels+1), number_of_channels, config=bp_configs.config_defaults, \
                                              depth=bp_configs.config_defaults['depth'], training=True, clm=False)
            # unet3p_dsv = enable_dropout(unet3p_dsv)            
            unet3p_dsv.load_weights(bp_configs.prod_dir + '/extras/enemy/' + in_string + '/variables/variables').expect_partial()
            outputs = unet3p_dsv.predict(np.array(test_data)[:,:,:,:number_of_channels+1], verbose=1, batch_size=1)[0]

            for i in range(number_of_channels):
                val = outputs[:,:,:,i]
                val[val<-0.5] = -1.0
                outputs[:,:,:,i] = val

            rand_indxs = random.sample(range(1, outputs.shape[0]), 10)
            maes = []
            for i in range(np.array(test_data).shape[0]):
                maes.append(calc_MAE(bp_utility.inv_standardize(test_channels[i,:16,:,0], 'ref', 'kazr'), bp_utility.inv_standardize(outputs[i,:16,:,0], 'ref', 'kazr')))
            
            final_maes.append(np.nanmean(maes))

    np.save(bp_configs.prod_dir + '/extras/enemy/short_' +  in_string + '.npy', final_maes)

    plt.show()

plotSHAP()
# getSHAPValues()
# performSHAP()

# cases = [('radar', 'temp'), ('radar', 'humidity'), ('radar', 'u_wind'), \
#         ('radar', 'v_wind'), ('radar', 'temp', 'humidity'), ('radar', 'temp', 'u_wind'), \
#         ('radar', 'temp', 'v_wind'), ('radar', 'humidity', 'u_wind'), \
#         ('radar', 'humidity', 'v_wind'), ('radar', 'u_wind', 'v_wind'), \
#         ('radar', 'temp', 'humidity', 'u_wind'), ('radar', 'temp', 'humidity', 'v_wind'), \
#         ('radar', 'temp', 'u_wind', 'v_wind'), ('radar', 'humidity', 'u_wind', 'v_wind'), \
#         ('radar', 'temp', 'humidity', 'u_wind', 'v_wind')]
# for case in cases:
#     shapCheckSummer(case)
