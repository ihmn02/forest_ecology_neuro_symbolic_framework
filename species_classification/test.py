# TO DO 
# modify model to use network with both x and chm arguments

import numpy as np
import cv2
from math import floor, ceil
import tqdm
from joblib import dump, load
from paths import data_path

import torch

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from os.path import join

from paths import *

def apply_pca(x, pca):
  N,H,W,C = x.shape
  x = np.reshape(x,(-1,C))
  x = pca.transform(x)
  x = np.reshape(x,(-1,H,W,x.shape[-1]))
  return x

def canopy_test(img_type, model):
    #pca = load(join(data_path, 'pca.joblib'))
    # "no data value" for labels
    label_ndv = 255

    # radius of square patch (side of patch = 2*radius+1)
    patch_radius = 7

    # height threshold for CHM -- pixels at or below this height will be discarded
    height_threshold = 5

    # tile size for processing
    tile_size = 128

    # tile size with padding
    padded_tile_size = tile_size + 2*patch_radius

    # open the hyperspectral or RGB image
    if (img_type == "rgb"):
      image = rasterio.open(rgb_image_uri)
    else:
        image = rasterio.open(image_uri)
    image_meta = image.meta.copy()
    image_ndv = image.meta['nodata']
    image_width = image.meta['width']
    image_height = image.meta['height']
    image_channels = image.meta['count']

    # load model
    input_shape = (padded_tile_size,padded_tile_size, 3) #(padded_tile_size,padded_tile_size, pca.n_components_)

    # calculate number of tiles
    num_tiles_y = ceil(image_height / float(tile_size))
    num_tiles_x = ceil(image_width / float(tile_size))

    print('Metadata for image')
    for key in image_meta.keys():
      print('%s:'%key)
      print(image_meta[key])
      print()

    # create predicted label raster
    predict_meta = image_meta.copy()
    predict_meta['dtype'] = 'uint8'
    predict_meta['nodata'] = label_ndv
    predict_meta['count'] = 1
    predict = rasterio.open("./" + '/' + predict_uri, 'w', compress='lzw', **predict_meta)

    # open the CHM
    chm = rasterio.open(chm_uri)
    chm_vrt = WarpedVRT(chm, crs=image.meta['crs'], transform=image.meta['transform'], width=image.meta['width'], height=image.meta['height'], resampling=Resampling.bilinear)

    # dilation kernel
    kernel = np.ones((patch_radius*2+1,patch_radius*2+1),dtype=np.uint8)

    # go through all tiles of input image
    # run convolutional model on tile
    # write labels to output label raster
    with tqdm.tqdm(total=num_tiles_y*num_tiles_x) as pbar:
        for y in range(patch_radius,image_height-patch_radius,tile_size):
            for x in range(patch_radius,image_width-patch_radius,tile_size):
                pbar.update(1)

                window = Window(x-patch_radius,y-patch_radius,padded_tile_size,padded_tile_size)

                # get tile from chm
                chm_tile = chm_vrt.read(1,window=window)
                if chm_tile.shape[0] != padded_tile_size or chm_tile.shape[1] != padded_tile_size:
                  pad = ((0,padded_tile_size-chm_tile.shape[0]),(0,padded_tile_size-chm_tile.shape[1]))
                  chm_tile = np.pad(chm_tile,pad,mode='constant',constant_values=0)

                chm_tile = np.expand_dims(chm_tile,axis=0)
                chm_bad = chm_tile <= height_threshold

                # get tile from image
                image_tile = image.read(window=window)
                image_pad_y = padded_tile_size-image_tile.shape[1]
                image_pad_x = padded_tile_size-image_tile.shape[2]
                output_window = Window(x,y,tile_size-image_pad_x,tile_size-image_pad_y)
                if image_tile.shape[1] != padded_tile_size or image_tile.shape[2] != padded_tile_size:
                  pad = ((0,0),(0,image_pad_y),(0,image_pad_x))
                  image_tile = np.pad(image_tile,pad,mode='constant',constant_values=-1)

                # re-order image tile to have height,width,channels
                image_tile = np.transpose(image_tile,axes=[1,2,0])

                # add batch axis
                image_tile = np.expand_dims(image_tile,axis=0)
                image_bad = np.any(image_tile < 0,axis=-1)

                image_tile = image_tile.astype('float32')
                #image_tile = apply_pca(image_tile, pca)

                # run tile through network
                #predict_tile = np.argmax(model(image_tile),axis=-1).astype('uint8')
                
                tile_as_tensor = torch.from_numpy(np.transpose(image_tile, [0, 3, 1, 2])).to(model.device)
                dummy_chm = torch.rand(tile_as_tensor.shape, dtype=torch.float32)
                dummy_dem = torch.rand(tile_as_tensor.shape, dtype=torch.float32)
                predict_tile = np.argmax(model(tile_as_tensor, dummy_chm, dummy_dem).permute(0, 2, 3, 1).cpu().detach().numpy(), axis=-1).astype('uint8')

                # dilate mask
                image_bad = cv2.dilate(image_bad.astype('uint8'),kernel).astype('bool')

                # set bad pixels to NDV
                predict_tile[chm_bad[:,patch_radius:-patch_radius,patch_radius:-patch_radius]] = label_ndv
                predict_tile[image_bad[:,patch_radius:-patch_radius,patch_radius:-patch_radius]] = label_ndv

                # undo padding
                if image_pad_y > 0:
                  predict_tile = predict_tile[:,:-image_pad_y,:]
                if image_pad_x > 0:
                  predict_tile = predict_tile[:,:,:-image_pad_x]

                # write to file
                predict.write(predict_tile,window=output_window)

    image.close()
    chm.close()
    predict.close()

if __name__ == '__main__':
    from os.path import join
    from sklearn.utils.class_weight import compute_class_weight

    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    from utils import Hsi_raster, trans
    from model import Net, Frickernet
    from ns_frickernet_pl import NsFrickerModel


    batch_size = args.batch
    n_trials = args.n_trials
    seed = args.seed


    train_ds = Hsi_raster(data_path, train_data_uri, transform=None, test=False, aux_data=args.aux_data)

    val_set_size = round(0.1 * len(train_ds) )
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_set_size, val_set_size], generator=torch.Generator().manual_seed(42))
    #train_ds, val_ds = random_split(train_ds, [len(train_ds) - 12004, 12004])
    #train_ds, val_ds, _  = random_split(train_ds, [1024, 512, len(train_ds)-1024-512], generator=torch.Generator().manual_seed(42))

    #val_ds = Hsi_raster(data_path, val_data_uri, transform=None, test=True, aux_data=args.aux_data)
    test_ds = Hsi_raster(data_path, test_data_uri, transform=None, test=True, aux_data=args.aux_data)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0)
    print("\nbaseline model?: {}".format(args.baseline))
    print("\nhyperparameter tune:? {}".format(args.htune))
    print("\ndata size: {}/{}/{}".format(len(train_ds), len(val_ds), len(test_ds)))
    print("\nbatch size: {}".format(args.batch))
    print("\nimage depth : {}".format(train_ds[0][0].shape[0]))
    print("\naux data: {}".format(args.aux_data))
    print("\nseed: {}".format(seed))
    print("\nn_trials: {}".format(args.n_trials))

    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(8), y=train_ds.dataset.y)
    print('\nclass weights: ', class_weights)
    class_weight_dict = {}
    for i in range(8):
        class_weight_dict[i] = class_weights[i]
    class_weight_tensor = torch.tensor(list(class_weight_dict.values()), dtype=torch.float32)
    #class_weight_tensor = None 

    n_layers = 6
    output_dim = 8
    init_num_filt=32
    max_num_filt=128
    img_depth= train_ds[0][0].shape[0]
    lr = args.lr
    scale = args.scale
    thr = args.thr
    lambdas = args.lambdas
    epochs = args.epochs
    wdir = args.wdir
    aux_data = args.aux_data


    model = NsFrickerModel(output_dim, img_depth, lambdas, scale, thr, init_num_filt=init_num_filt, max_num_filt=max_num_filt, num_layers=n_layers, lr=lr, class_wts=class_weight_tensor)
    #model = NsFrickerModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/exp-epoch04-val-f1_0.77.ckpt")
    checkpoint = torch.load("lightning_logs/version_0/checkpoints/exp-epoch04-val-f1_0.77.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    canopy_test("rgb", model)
