import torch
from torchvision import transforms
from torch.utils.data import Dataset
import h5py as h5
from os.path import join
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
import albumentations as A
from tqdm import tqdm, trange
from pars import args

trans = A.Compose(
    [A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5)],
    additional_targets={'chm': 'image', 'dem': 'image'}
)


class Hsi_raster(Dataset):
    def __init__(self, wdir, h5_filename, transform=None, test=False, aux_data=None):
        """
                Args:
                    h5_filename (string): Path to the hd5 file with rasters and annotations
                    wdir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                """

        self.wdir = wdir
        self.test = test
        h5_file = h5.File(join(wdir, h5_filename), "r")
        self.y = np.array(h5_file['label'])
        self.chm = np.array(h5_file['chm'], dtype=np.float32)

        # performance improves when dem same magnitude as image data
        if args.rgb:
           k_dem = 0.1
        else:
           k_dem = 0.01
        self.dem = k_dem * np.array(h5_file['dem'], dtype=np.float32)

        self.x = np.array(h5_file['data'], dtype=np.float32)
        if (test == False) and (self.x.shape[-1] >= 32):
            self.pca = self.estimate_pca(self.x)
            self.x = self.apply_pca(self.x, self.pca)
        elif (test == True) and (self.x.shape[-1] >= 32):
            self.pca = load(join(wdir, 'pca.joblib'))
            self.x = self.apply_pca(self.x, self.pca)
            
        if self.test == False:
            self.x, self.y = self.augment_images(self.x, self.y)
            self.chm, _ = self.augment_images(self.chm, self.y)
            self.dem, _ = self.augment_images(self.dem, self.y)
        
        # add chm or dem to rgb raster as an extra channel
        if aux_data == "chm":
            self.x = self.stack_data(self.x, self.chm)
        elif aux_data == "dem":
            self.x = self.stack_data(self.x, self.dem)

        self.transform = transform

    def __len__(self):
        length = self.x.shape[0]
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        chm = self.chm[idx] 
        dem = self.dem[idx]
        if (self.transform is not None):
            transformed = self.transform(image=x, chm=chm, dem=dem)
            x_tensor = torch.from_numpy(np.transpose(transformed['image'], axes=[2, 0, 1]))
            chm_tensor =  torch.unsqueeze(torch.from_numpy(transformed['chm']), 0)
            dem_tensor = torch.unsqueeze(torch.from_numpy(transformed['dem']), 0)
        else:
            x = np.transpose(x, axes=[2, 0, 1])
            x_tensor = torch.from_numpy(x)
            chm_tensor = torch.unsqueeze(torch.from_numpy(chm), 0)
            dem_tensor = torch.unsqueeze(torch.from_numpy(dem), 0)

        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)

        return x_tensor, chm_tensor, dem_tensor, y_tensor

    def estimate_pca(self, x):
        x_sample = x[:, 7, 7]
        pca = PCA(32, whiten=True, random_state=0)
        pca.fit(x_sample)
        if self.test == False:
            dump(pca, join(self.wdir, 'pca.joblib'))
        return pca

    def apply_pca(self, s, pca):
        N, H, W, C = s.shape
        t = np.reshape(s, (-1, C))
        t = pca.transform(t)
        t = np.reshape(t, (-1, H, W, t.shape[-1]))
        return t

    def augment_images(self, x, y):
        x_aug = []
        y_aug = []
        with tqdm(total=len(x) * 8, desc='augmenting images') as pbar:
            for rot in range(4):
                for flip in range(2):
                    for patch, label in zip(x, y):
                        patch = np.rot90(patch, rot)
                        if flip:
                            patch = np.flip(patch, axis=0)
                            patch = np.flip(patch, axis=1)
                        x_aug.append(patch)
                        y_aug.append(label)
                        pbar.update(1)
        return np.stack(x_aug, axis=0), np.stack(y_aug, axis=0)

    def stack_data(self, x, data):
        data = np.where(data == -9999, -1, data) #-9999 is no data; chm range -1 to 63.02
        data = np.expand_dims(data, axis=3)
        x = np.concatenate([x, data], axis=3)
        return x


