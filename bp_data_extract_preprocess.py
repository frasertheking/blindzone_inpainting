#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fraser king
@description: Data preprocessing script adapted from Geiss and Hardin 2021
"""

import os
import numpy as np
import bp_configs
import bp_utility
import bp_plotting
import bp_batch
from datetime import datetime, timezone
from glob import glob
from netCDF4 import Dataset
from multiprocess import Pool
from scipy.interpolate import UnivariateSpline

def path_builder():
    data_paths = []
    for site in bp_configs.SITES_TO_EXAMINE:
        for year in bp_configs.YEARS_TO_EXAMINE:
            for month in bp_configs.MONTHS_TO_EXAMINE:
                data_paths.append(site + '/' + year + '/' + month)
    return data_paths


#   DATA PREPROCESSING   #####################################################
def ingest_kazr(data_paths):
    nc_dir = bp_configs.raw_data_dir
    era5_nc_dir = bp_configs.raw_era5_data_dir
    met_nc_dir = bp_configs.raw_met_data_dir
    out_dir = bp_configs.out_dir

    def scale(im, nR, nC):
        nR0 = len(im)
        nC0 = len(im[0])
        return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
                    for c in range(nC)] for r in range(nR)]

    def proc_nc_file(f):
        print('Processing data from: ' + f)
        f2 = glob(era5_nc_dir + f[len(nc_dir):len(nc_dir)-1 + 12] + '/*.nc')[0]
        site = f[len(nc_dir):len(nc_dir)-1 + 4]

        num_bins = 333 # was 256, 333 refers to 10 km above surface

        dset = Dataset(f)
        base_time = dset['base_time'][:]
        time_obj = datetime.fromtimestamp(base_time, timezone.utc)
        starting_hour = (time_obj.day - 1) * 24 + time_obj.hour

        lat_idx = bp_configs.SITE_LAT_LON_IDX[site][0]
        lon_idx = bp_configs.SITE_LAT_LON_IDX[site][1]

        era5_ds = Dataset(f2)
        t = era5_ds['t'][:].data.T[lat_idx][lon_idx]
        q = era5_ds['q'][:].data.T[lat_idx][lon_idx]
        u = era5_ds['u'][:].data.T[lat_idx][lon_idx]
        v = era5_ds['v'][:].data.T[lat_idx][lon_idx]

        x = dset['reflectivity_best_estimate'][:].data.T[:num_bins,:]
        x = np.float16(x)
        ref = bp_utility.standardize(x,'ref','kazr')
        mask = ref<-0.5
        scaled_ref = np.asarray(scale(ref, 128, 21600))

        precip = dset['precip_mean'][:].data
        
        site_w_short = 'nsametC1.b1.'
        if site == 'oli':
            site_w_short = 'olimetM1.b1.'
        
        f3 = met_nc_dir + '/' + site_w_short + str(time_obj.year) + str(time_obj.month).zfill(2) + str(time_obj.day).zfill(2) + '.000000.cdf'
        expanded_temp = np.full(21600, np.nan)
        if os.path.isfile(f3):
            t_ds = Dataset(f3)
            expanded_temp = np.repeat(t_ds['temp_mean'][:].data, 15)

        weather = x>bp_configs.DATA_RANGE['kazr']['ref'][0]
        w_frac = np.mean(np.double(weather))
        l_count = np.sum(weather[:,:2])
        r_count = np.sum(weather[:,-2:])
        if w_frac<0.01 and l_count<10 and r_count<10:
            return None

        if bp_configs.USE_DOP_SPW:
            x = dset['mean_doppler_velocity'][:].data.T[:num_bins,:]
            x = np.double(x)
            x[mask] = 0.0
            x[np.abs(x)>bp_configs.NYQ['kazr']] = 0.0
            x = bp_utility.standardize(x,'vel','kazr')
            vel = np.float16(x)
            scaled_vel = np.asarray(scale(vel, 128, 21600))
            
            x = dset['spectral_width'][:].data.T[:num_bins,:]
            x = np.float16(x)
            x = bp_utility.standardize(x,'wid','kazr')
            x[mask] = -1.0
            wid = x
            scaled_wid = np.asarray(scale(wid, 128, 21600))
            return np.stack((scaled_ref,scaled_vel,scaled_wid),axis=-1)[:,:,:]
        else:
            levels = 20 #18
            old_indices = np.arange(levels)
            new_length = 128
            new_indices = np.linspace(0,levels,new_length)
            t_combined = []
            q_combined = []
            u_combined = []
            v_combined = []
            for i in range(24):
                for j in range(7):
                    for k in range(128):
                        loc_t = t.T[starting_hour+i][::-1][0:levels]
                        spl = UnivariateSpline(old_indices,loc_t,k=3,s=0)
                        t_combined.append(spl(new_indices))

                        loc_q = q.T[starting_hour+i][::-1][0:levels]
                        spl = UnivariateSpline(old_indices,loc_q,k=3,s=0)
                        q_combined.append(spl(new_indices))

                        loc_u = u.T[starting_hour+i][::-1][0:levels]
                        spl = UnivariateSpline(old_indices,loc_u,k=3,s=0)
                        u_combined.append(spl(new_indices))

                        loc_v = v.T[starting_hour+i][::-1][0:levels]
                        spl = UnivariateSpline(old_indices,loc_v,k=3,s=0)
                        v_combined.append(spl(new_indices))
            t_combined = np.asarray(t_combined).T
            q_combined = np.asarray(q_combined).T
            u_combined = np.asarray(u_combined).T
            v_combined = np.asarray(v_combined).T

            t_combined = np.float16(t_combined)
            t_combined = bp_utility.standardize(t_combined,'t','kazr')
            q_combined = np.float16(q_combined)
            q_combined = bp_utility.standardize(q_combined,'q','kazr')
            u_combined = np.float16(u_combined)
            u_combined = bp_utility.standardize(u_combined,'u','kazr')
            v_combined = np.float16(v_combined)
            v_combined = bp_utility.standardize(v_combined,'v','kazr')

            scaled_ref = scaled_ref[:,-t_combined.shape[1]:]
            precip = precip[-t_combined.shape[1]:]
            expanded_temp = expanded_temp[-t_combined.shape[1]:]
            return (np.stack((scaled_ref,t_combined,q_combined,u_combined,v_combined),axis=-1)[:,:,:], precip, expanded_temp)
    
    for path in data_paths:
        print("Working on", path)
        print(nc_dir + path + '/*.nc')

        files = glob(nc_dir + path + '/*.nc')
        if len(files) == 0:
            print("No files for this station/year/month...")
            continue
        files.sort()

        p = Pool(24)
        data = p.map(proc_nc_file,files,chunksize=1)
        p.close()

        data2 = []
        data3 = []
        data4 = []
        for d in data:
            if not d is None:
                data2.append(d[0])
                data3.append(d[1])
                data4.append(d[2])

        data = np.concatenate(data2, axis=1)
        precip_data = np.concatenate(data3, axis=0)
        temp_data = np.concatenate(data4, axis=0)
        data = np.concatenate(data2, axis=1)
        station_name = path.split('/')[0]
        station_year_tmp = path.split('/')[1]
        month = path.split('/')[2]

        np.save(out_dir + '/preprocessed/' + station_name + '_' + station_year_tmp + '_' + month + '_kazr.npy', data)
        # np.save(out_dir + '/preprocessed/' + station_name + '_' + station_year_tmp + '_' + month + '_precip.npy', precip_data)
        np.save(out_dir + '/preprocessed_met/' + station_name + '_' + station_year_tmp + '_' + month + '_temp.npy', temp_data)

def get_kazr_sample_inds():
    files = glob(bp_configs.out_dir + '/preprocessed/*.npy')
    files.sort()

    for file in files:
        print('Loading Data...', file)
        mask = np.load(file)[:,:,0]
        mask = np.float16(mask>-1.0)

        print('Computing Convolution...')
        mask = bp_utility.boxcar2d(mask,bp_configs.min_weather_size)>0.99
        
        #the indices for the downfilling cases:
        #enforce that the bottom 1/4 of the sample has weather
        print('Computing Downfilling Mask...')
        downfill_mask = np.any(mask[bp_configs.DOWNFILL_CUT_RANGE[0]:,:],axis=0) # SPONGE: change to 1 if multiple cut lines
        downfill_mask = bp_utility.boxcar1d(np.float16(downfill_mask[:,np.newaxis]),bp_configs.SIZE['downfill'][1])
        downfill_inds = np.where(downfill_mask[:,0]>0)[0] #SPONGE: NOTE THAT THIS IS CHANGED TO >= TO INCLUDE AREAS WITH NO WEATHER
        np.save(bp_configs.out_dir + '/indices/indices_' + os.path.basename(file), downfill_inds)
  
#gets the test dataset
def make_kazr_test_sets():
    files = glob(bp_configs.out_dir + '/preprocessed/*.npy')
    files.sort()

    for file in files:
        print('Loading Data...', os.path.basename(file))
        data = np.load(bp_configs.out_dir + '/preprocessed/' + os.path.basename(file))
        data_temp = np.load(bp_configs.out_dir + '/preprocessed_met/' + os.path.basename(file)[:11] + '_temp.npy')
        inds = np.load(bp_configs.out_dir + '/indices/indices_' + os.path.basename(file))
        inds = inds[int(-len(inds)*bp_configs.TEST_FRAC):]
        test_set = []
        test_set_temps = []
        N = bp_configs.SIZE['downfill'][0]
        last = -N
        print("TEST SET SIZE:", N, np.asarray(test_set).shape)
        for ind in inds:
            if ind-last > N//2:
                test_set.append(data[:,ind:ind+N,:])
                test_set_temps.append(data_temp[ind:ind+N])
                last = ind
        np.save(bp_configs.out_dir + '/test_set/test_set_' + os.path.basename(file), test_set)
        np.save(bp_configs.out_dir + '/test_set/test_set_' + os.path.basename(file)[:11] + '_temp.npy', test_set_temps)

def create_sample_sets(case):
    size = bp_configs.SIZE[case]
    files = glob(bp_configs.out_dir + '/preprocessed/*.npy')
    files.sort()

    samples_x = []
    x = np.zeros((2,*size,bp_configs.CHANNELS+1))
    for file in files:
        print('Loading Data...', os.path.basename(file))
        data = np.load(bp_configs.out_dir + '/preprocessed/' + os.path.basename(file))
        inds = np.load(bp_configs.out_dir + '/indices/indices_' + os.path.basename(file))

        if bp_configs.CHANNELS == 1:
            data = np.expand_dims(data[:,:,bp_configs.CHANNELS-1], axis=2)
        batch = bp_batch.BATCH_FUNC[case]
        DOWNFILL_SAMPLE_INDS = inds = inds[int(-len(inds)*bp_configs.TEST_FRAC):]

        while(True):
            batch(x,data,DOWNFILL_SAMPLE_INDS)
            bp_plotting.plot(x[0,:,:,:5],case)
            bp_plotting.show()
            response = input('Add to sample set? (y)')
            if response == 'y':
                samples_x.append(np.copy(x[0,...]))
                bp_plotting.plot(x[0,:,:,:5],case,fname=bp_configs.out_dir + "/samples/images/" + os.path.splitext(os.path.basename(file))[0] + "_sample.png")
                break
        
    np.save(bp_configs.out_dir + '/samples/all_samples.npy', np.array(samples_x))

#### Main runloop
if __name__ == '__main__':
    data_paths = path_builder()
    print("Loading data from:", data_paths)

    # print("\nspinning up")
    # ingest_kazr(data_paths)

    # print("\nGetting sample indices")
    # get_kazr_sample_inds()

    print("\nMaking test sets")
    make_kazr_test_sets()


    # print("\nCreating cases")
    # create_sample_sets('downfill')