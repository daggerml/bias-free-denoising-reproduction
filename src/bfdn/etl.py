from bfdn.util import RAW_DATA_PATH, DATA_PATH
from metaflow import FlowSpec, step
from metaflow import Parameter
import numpy as np


def Im2Patch(img, win, stride=1):
    "very slightly modified from original"
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:,
                0: endw - win + 0 + 1: stride,
                0: endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,
                        i: endw - win + i + 1: stride,
                        j: endh - win + j + 1: stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def _proc_img(rng, img_path, scale_floats, patch_size,
              stride, aug_times):
    # FIXME select distinct transformations (choice not randint)
    from PIL import Image
    import numpy as np
    raw_img = Image.open(img_path)
    h, w = raw_img.size
    for k, v in enumerate(scale_floats):
        img = raw_img.resize((int(h * v), int(w * v)), resample=3)
        img = np.array(img)
        img = np.expand_dims(img[:, :].copy(), 0) / 255.0
        patches = Im2Patch(img, win=patch_size, stride=stride)
        for i in range(patches.shape[3]):
            data = patches[:, :, :, i]
            yield data.copy()
            for m in range(aug_times - 1):
                data_aug = data_augmentation(data, rng.integers(1, 8))
                yield data_aug.copy()
    return


class ETLFlow(FlowSpec):
    data_path = Parameter(
        "data-path",
        help="",
        type=str,
        default=RAW_DATA_PATH,
    )
    out_path = Parameter(
        "out-path",
        help="",
        type=str,
        default=DATA_PATH,
    )
    scales = Parameter(
        "scales",
        help="",
        default="1,0.9,0.8,0.7",
        type=str,
    )
    patch_size = Parameter(
        "patch-size",
        help="",
        default=50,
        type=int,
    )
    stride = Parameter(
        "stride",
        help="",
        default=10,
        type=int,
    )
    aug_times = Parameter(
        "aug-times",
        help="",
        default=2,
        type=int,
    )
    seed = Parameter(
        "seed",
        help="",
        default=42,
        type=int,
    )
    debug = Parameter(
        "debug",
        help="",
        default=False,
        type=bool,
    )

    @step
    def start(self):
        print(f" data path: {self.data_path}")
        self.scale_floats = [float(sc) for sc in self.scales.split(",")]
        self.next(self.build_train)

    @step
    def build_train(self):
        from glob import glob
        from h5py import File
        import numpy as np
        from tqdm import tqdm
        from tqdm.contrib.logging import tqdm_logging_redirect
        files_path = f'{self.data_path}/Train400/*.png'
        files = sorted(glob(files_path))
        rng = np.random.default_rng(self.seed)
        with tqdm_logging_redirect():
            with File(f'{self.out_path}/train.h5', 'w') as h5f:
                for i, img_file in enumerate(tqdm(files, desc='train')):
                    for j, img in enumerate(_proc_img(rng, img_path=img_file,
                                                      scale_floats=self.scale_floats,
                                                      patch_size=self.patch_size,
                                                      stride=self.stride,
                                                      aug_times=self.aug_times)):
                        if i == 0:
                            j_max = j + 1
                        n = i * j_max + j
                        h5f.create_dataset(str(n), data=img)
        self.next(self.build_valid)

    @step
    def build_valid(self):
        from glob import glob
        from h5py import File
        from PIL import Image
        from tqdm import tqdm
        from tqdm.contrib.logging import tqdm_logging_redirect
        files_path = f'{self.data_path}/Test/Set12/*.png'
        files = sorted(glob(files_path))
        with tqdm_logging_redirect():
            with File(f'{self.out_path}/valid.h5', 'w') as h5f:
                for i, img_file in enumerate(tqdm(files, desc='valid')):
                    img = Image.open(img_file)
                    img = np.expand_dims(np.array(img), 0) / 255.0
                    h5f.create_dataset(str(i), data=img)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    ETLFlow()
