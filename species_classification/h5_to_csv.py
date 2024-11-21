"""
This script pulls counts and tree heights from datasets
"""

import h5py as h5
import numpy as np
import pandas as pd
from os.path import join

wdir = "./data"
train_uri = "train_data.hdf5"
test_uri = "test_data.hdf5"

fold_basename = "ns_rgb_fold"

uri_list = [train_uri, test_uri]

def process_chm(chm):
    #max_ht = [np.mean(x[x>0]) for x in chm]
    max_ht = [np.max(x) for x in chm]
    return max_ht

def process_dem(dem):
    max_elev = [np.max(x) for x in dem]
    return max_elev

def process_r_ratio(rgb_data):
    patch_sum = np.sum(np.sum(rgb_data, axis=1), axis=1)
    r_ratio = patch_sum[:, 1] / np.sum(patch_sum, axis=1)
    return r_ratio

def process_gli(rgb_data):
    patch_sum = np.sum(np.sum(rgb_data[:, 5:8, 5:8, :], axis=1), axis=1)

    #gli = (2G - R - B) / (2G + R + B)
    gli = ((2.0 * patch_sum[:, 1]) - patch_sum[:, 0] - patch_sum[:, 2]) / ((2.0 * patch_sum[:, 1]) + patch_sum[:, 0] + patch_sum[:, 2])
    return gli    

def append_to_df(df, fold_num, ds_uri, chm_arr, dem_arr, r_ratio, gli_arr, label_arr):
    fold_arr = [fold_num] * len(chm_arr)
    ds_uri_arr = [ds_uri.split("_")[0]] * len(chm_arr)

    temp_df = pd.DataFrame({"fold":fold_arr, "dataset":ds_uri_arr, "max_ht":chm_arr, "max_elev":dem_arr, "r_ratio":r_ratio, "gli":gli_arr, "label":label_arr})
    if df is not None:
        df = pd.concat([df, temp_df], ignore_index=True)
    else:
        df = temp_df
    return df

def process_hsi(hsi, label):
    pass

df_list = []

for fold_num in range(10):
   res_df = None
   for uri in uri_list:
      fold_dir = fold_basename + str(fold_num)
      print("Reading from: {}".format(fold_dir))
      print("  Reading {}".format(join(wdir, fold_dir, uri)))
      h5file = h5.File(join(wdir, fold_dir, uri), "r")

      print("  Getting CHM...")
      label_arr = np.array(h5file['label'])
      chm = np.array(h5file['chm'])
      chm_arr = process_chm(chm)

      # process DEM
      print("  Getting DEM...")
      dem = np.array(h5file['dem'])
      dem_arr = process_dem(dem)

      # process r_ratio
      print("  Getting R-Ratio...")      
      r_ratio = process_r_ratio(h5file['data'])

      # process gli
      print("  Getting GLI...")
      gli = process_gli(h5file['data'])

      # grow dataframe
      res_df = append_to_df(res_df, fold_num, uri, chm_arr, dem_arr, r_ratio, gli, label_arr)
   df_list.append(res_df)

all_folds_df = pd.concat(df_list)
import pdb; pdb.set_trace()
all_folds_df.to_csv("rgb_all_folds4.csv", index=False, header=True)
