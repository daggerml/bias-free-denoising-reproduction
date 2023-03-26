import h5py
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from bfdn.etl import DATA_PATH


class Data(Dataset):
    def __init__(self, train=True, max_keys=-1):
        super(Dataset, self).__init__()
        self.train = train
        fname = 'train' if train else 'valid'
        self.fname = f'{DATA_PATH}/{fname}.h5'
        with h5py.File(self.fname, 'r') as f:
            keys = sorted(list(f.keys()))
        if max_keys > 0:
            keys = keys[:max_keys]
        self.max_keys = max_keys
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        key = self.keys[index]
        with h5py.File(self.fname, 'r') as f:
            data = np.array(f[key], dtype=np.float32)
        return tensor(data)
