import numpy as np
import h5py
from torch.utils.data import Dataset

class audio_dataset(Dataset):
    def __init__(self, data_path, yaml_path, val_fold, train = True):
        self.data_path = data_path
        self.val_fold = val_fold
        self.mean, self.std = self.calculate_mean_var()

        # Load h5 data
        hf = h5py.File(self.data_path, 'r')
        fold = np.array(hf['fold'])
        self.foldidx = np.argwhere(fold != self.val_fold).reshape(-1) if train else np.argwhere(fold == valfold).reshape(-1)
        self.foldidxlist = self.foldidx.tolist()
        self.audio_names = np.array([s.decode() for s in hf['audio_name'][self.foldidxlist]])
        self.x = hf['mixture_logmel'][self.foldidxlist]
        self.y = hf['target'][self.foldidxlist].astype('float32')
        self.folds = hf['fold'][self.foldidxlist]
        hf.close()

        # z score standardization
        self.x = (self.x - self.mean) / self.std

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.audio_names[index]

    def __len__(self):
        return len(self.x)

    def calculate_mean_var(self):
        hf = h5py.File(self.data_path, 'r')
        fold = np.array(hf['fold'])
        foldidx = np.argwhere(fold != self.val_fold).reshape(-1) # stored in form (6000, )
        foldidxlist = foldidx.tolist() 
        x = hf['mixture_logmel'][foldidxlist]
        if x.ndim <= 2:
            axis = 0
        elif x.ndim == 3:
            axis = (0, 1)
        mean = np.mean(x, axis=axis)
        std = np.std(x, axis=axis)
        hf.close()
        return mean, std



