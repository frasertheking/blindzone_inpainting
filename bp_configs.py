#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Configuration file that controls the other project modules
"""

# These are the main project configuration values you can tweak as necessary
CHANNELS = 1
USE_DOP_SPW = False
RUN_CASE='128_' + str(CHANNELS) + 'chan_era5_nsa_oli_10km_dsv_long_hybrid'
RUN_CASE_DATA='128_5chan_era5_nsa_oli_10km_dsv_long_hybrid/'
RUN_NAME='test'
DOWNFILL_SIZES = [16] #[20, 40, 60]
N_MC_TESTS = 50 # monte carlo dropout tests numbers
STD_CUTOFF = 1
PERC_CLOUD = 0.05
CLOUD_MASK = -0.5
PRECIPITATION_DBZ = -20
MAX_DBZ = 20
BUF_RANGE = [1,10]
min_weather_size = 10
DOWNFILL_CUT_RANGE=[16]
TEST_FRAC = 0.1
NYQ = {'kazr':8.0}
SIZE = {'downfill': [128,128]}
DATA_RANGE = {'kazr': {'ref': [-60,30], 't': [180, 300], 'q':[5.7e-08,0.0115], 'u': [-30, 30], 'v': [-30, 30], 'vel': [-NYQ['kazr']*1.5,NYQ['kazr']*1.5], 'wid':[0,2.5]}}
SITE_LAT_LON_IDX = {'nsa': [2, 3], 'oli': [2, 3], 'awr': [4, 2]}

# tuned configuration defaults for the unet3p models
config_defaults = {
    'depth': 4,
    'batch_size': 16,
    'epochs': 501,
    'lr': 5e-5,
    'l2_reg': 1e-5,
    'filters': 32,
    'kernel_size': 3,
    'dropout': 0.2,
    'optimizer': 'adam',
    'interpolation': 'bilinear',
    'momentum': 0.9,
    'seed': 42
}

#### Windows
project_path = 'D:/Development/blindpaint/data/'
image_path = 'D:/Development/blindpaint/figures/2_site_comparisons/'
model_dir = 'D:/Development/blindpaint/data/models/'
raw_data_dir = 'Z:/data/just_for_transferring_dont_code_here/blindpaint/data/kazr/' #project_path + 'raw_data/'
raw_era5_data_dir = 'Z:/data/just_for_transferring_dont_code_here/blindpaint/data/era5/' #project_path + 'raw_data/'
raw_met_data_dir = 'Z:/data/just_for_transferring_dont_code_here/blindpaint/data/met/' #project_path + 'raw_data/'

#### UM Mac
# project_path = '/Users/kingfr/Development/blindpaint/data/'
# image_path = '/Users/kingfr/Development/blindpaint/data/figures/'
# model_dir = '/Users/kingfr/Development/blindpaint/data/models/'
# raw_data_dir = '/Users/kingfr/Development/blindpaint/data/kazr/' #project_path + 'raw_data/'
# raw_era5_data_dir = '/Users/kingfr/Development/blindpaint/data/era5/' #project_path + 'raw_data/'

#### Azure
# project_path = '/datadrive/' #'/Users/kingfr/Development/blindpaint/data/'
# image_path = '/datadrive/figures/2_site_comparisons/'
# model_dir ='/datadrive/models/' #'/Users/kingfr/Development/blindpaint/data/models/'
# raw_data_dir = project_path + 'kazr/' #'/Users/kingfr/Development/blindpaint/data/kazr/'
# raw_era5_data_dir = project_path + 'era5/'
# raw_met_data_dir = project_path + 'met/'

out_dir = project_path + 'out/' + RUN_CASE
data_dir = project_path + 'out/' + RUN_CASE_DATA
eval_dir = project_path + 'eval/' + RUN_CASE
prod_dir = project_path + 'prod/'

SITES_TO_EXAMINE = ['nsa']#, 'oli']
YEARS_TO_EXAMINE =  ['2016']#, '2013', '2014', '2015', '2016', '2017']
MONTHS_TO_EXAMINE = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

FINAL_PERIODS = ['nsa_2012_03', 'nsa_2012_04', 'nsa_2012_11', 'nsa_2012_12', 'nsa_2013_01', \
                 'nsa_2013_10', 'nsa_2013_11', 'nsa_2013_12', 'nsa_2014_01', 'nsa_2014_02', 'nsa_2014_03', \
                 'nsa_2014_04', 'nsa_2014_10', 'nsa_2014_11', 'nsa_2014_12', 'nsa_2015_01', 'nsa_2015_02', \
                 'nsa_2015_03', 'nsa_2015_04', 'nsa_2016_01', 'nsa_2016_02', 'nsa_2016_03', 'nsa_2016_04', \
                 'nsa_2016_11', 'nsa_2016_12', 'nsa_2017_03', 'nsa_2017_04', 'oli_2015_11', \
                 'oli_2015_12', 'oli_2016_01', 'oli_2016_02', 'oli_2016_03', 'oli_2016_11', 'oli_2016_12', \
                 'oli_2017_01', 'oli_2017_02', 'oli_2017_03', 'oli_2017_04', 'oli_2017_11']

# SITES_TO_EXAMINE = ['nsa', 'oli']
# YEARS_TO_EXAMINE =  ['2012', '2013', '2014', '2015', '2016', '2017']
# MONTHS_TO_EXAMINE = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
